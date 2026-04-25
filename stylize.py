import os
import time
from collections import OrderedDict
from functools import reduce

import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

try:
    from absl import logging as absl_logging  # type: ignore

    absl_logging.set_verbosity(absl_logging.ERROR)
except Exception:
    pass

import numpy as np
import tensorflow as tf
from PIL import Image

import vgg

# Silence TensorFlow logs as early as possible (must run before graph building).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

tf.compat.v1.disable_v2_behavior()

try:
    tf.get_logger().setLevel("ERROR")
except Exception:
    pass
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

CONTENT_LAYERS = ("relu4_2", "relu5_2")
STYLE_LAYERS = ("relu1_1", "relu2_1", "relu3_1", "relu4_1", "relu5_1")


def _resize_mask_to_layer(mask_hw1: tf.Tensor, layer: tf.Tensor) -> tf.Tensor:
    """
    Resize a (1,H,W,1) mask to the spatial size of a VGG feature layer.
    """
    # layer shape is static in TF1 graphs here
    _, h, w, _ = layer.get_shape().as_list()
    return tf.image.resize(mask_hw1, size=[h, w], method=tf.image.ResizeMethod.BILINEAR)


def _masked_gram_matrix(feats: tf.Tensor, mask: tf.Tensor | None) -> tf.Tensor:
    """
    feats: (1,h,w,c). mask: (1,h,w,1) in [0,1] or None.
    Returns Gram matrix (c,c).
    """
    _, h, w, c = feats.get_shape().as_list()
    x = feats
    if mask is not None:
        x = x * mask
        norm = tf.reduce_sum(mask) * float(c) + 1e-8
    else:
        norm = float(h * w * c)

    x2 = tf.reshape(x, (-1, c))  # (h*w, c)
    gram = tf.matmul(x2, x2, transpose_a=True) / norm
    return gram


def _adain_stats(feats: tf.Tensor, mask: tf.Tensor | None) -> tuple[tf.Tensor, tf.Tensor]:
    """
    Compute per-channel mean/std for AdaIN-style losses.
    feats: (1,h,w,c). mask: (1,h,w,1) or None.
    Returns (mean(c,), std(c,)).
    """
    _, h, w, c = feats.get_shape().as_list()
    x = feats
    if mask is None:
        mu = tf.reduce_mean(x, axis=[1, 2])  # (1,c)
        var = tf.reduce_mean(tf.square(x - mu[:, None, None, :]), axis=[1, 2])
        sigma = tf.sqrt(var + 1e-8)
        return mu[0], sigma[0]

    # masked stats
    m = mask
    denom = tf.reduce_sum(m) + 1e-8
    mu = tf.reduce_sum(x * m, axis=[1, 2]) / denom  # (1,c)
    var = tf.reduce_sum(tf.square(x - mu[:, None, None, :]) * m, axis=[1, 2]) / denom
    sigma = tf.sqrt(var + 1e-8)
    return mu[0], sigma[0]


def _style_layer_loss(
    *,
    layer: tf.Tensor,
    style_target_gram: np.ndarray,
    style_target_stats: tuple[np.ndarray, np.ndarray],
    use_adain: bool,
    mask: tf.Tensor | None,
) -> tf.Tensor:
    """
    Style loss for a single layer.
    - Gram loss (default): L2 between Gram matrices.
    - AdaIN loss (optional): L2 between per-channel mean/std.
    """
    if use_adain:
        mu_t, sig_t = style_target_stats
        mu, sig = _adain_stats(layer, mask)
        mu_t = tf.constant(mu_t, dtype=tf.float32)
        sig_t = tf.constant(sig_t, dtype=tf.float32)
        return tf.reduce_mean(tf.square(mu - mu_t)) + tf.reduce_mean(tf.square(sig - sig_t))

    gram = _masked_gram_matrix(layer, mask)
    target = tf.constant(style_target_gram, dtype=tf.float32)
    return 2.0 * tf.nn.l2_loss(gram - target) / float(style_target_gram.size)


def _extract_style_patches_numpy(feats: np.ndarray, patch_size: int = 3, stride: int = 1) -> np.ndarray:
    """
    Extract normalized patch vectors from a style feature map.
    feats: (1,h,w,c)
    Returns (N, D) float32 where D = patch_size*patch_size*c.
    """
    _, h, w, c = feats.shape
    ps = int(patch_size)
    st = int(stride)
    patches = []
    for y in range(0, h - ps + 1, st):
        for x in range(0, w - ps + 1, st):
            p = feats[0, y : y + ps, x : x + ps, :].reshape(-1)
            patches.append(p)
    if not patches:
        return np.zeros((0, ps * ps * c), dtype=np.float32)
    mat = np.stack(patches, axis=0).astype(np.float32)
    # Normalize for cosine similarity
    mat /= (np.linalg.norm(mat, axis=1, keepdims=True) + 1e-8)
    return mat


def _patch_style_loss(layer: tf.Tensor, style_patch_matrix: tf.Tensor, mask: tf.Tensor | None) -> tf.Tensor:
    """
    Patch-based style loss using cosine similarity nearest-neighbor matching.
    - Extract patches from `layer` (relu3_1 by default)
    - For each content patch, find best matching style patch by cosine similarity
    - Loss = mean(1 - max_cos)

    This is optional and can be memory heavy for large feature maps.
    """
    ps = 3
    patches = tf.image.extract_patches(
        images=layer,
        sizes=[1, ps, ps, 1],
        strides=[1, 1, 1, 1],
        rates=[1, 1, 1, 1],
        padding="VALID",
    )  # (1, ph, pw, D)
    _, ph, pw, d = patches.get_shape().as_list()
    x = tf.reshape(patches, (-1, d))  # (N, D)
    x = tf.nn.l2_normalize(x, axis=1, epsilon=1e-8)

    if mask is not None:
        # downsample mask to patch grid by average pooling
        m = tf.image.resize(mask, size=[ph + ps - 1, pw + ps - 1], method=tf.image.ResizeMethod.BILINEAR)
        m = tf.nn.avg_pool(m, ksize=[1, ps, ps, 1], strides=[1, 1, 1, 1], padding="VALID")  # (1,ph,pw,1)
        m = tf.reshape(m, (-1, 1))
    else:
        m = None

    # Cosine similarity: (N, Ns)
    sim = tf.matmul(x, style_patch_matrix, transpose_b=True)
    best = tf.reduce_max(sim, axis=1, keepdims=True)  # (N,1)
    loss_vec = 1.0 - best
    if m is not None:
        denom = tf.reduce_sum(m) + 1e-8
        return tf.reduce_sum(loss_vec * m) / denom
    return tf.reduce_mean(loss_vec)


def get_loss_vals(loss_store):
    return OrderedDict((key, val.eval()) for key, val in loss_store.items())


def print_progress(loss_vals):
    for key, val in loss_vals.items():
        print("{:>13s} {:g}".format(key + " loss:", val))


def stylize(
    network,
    initial,
    initial_noiseblend,
    content,
    styles,
    preserve_colors,
    iterations,
    content_weight,
    content_weight_blend,
    style_weight,
    style_layer_weight_exp,
    style_blend_weights,
    tv_weight,
    learning_rate,
    beta1,
    beta2,
    epsilon,
    pooling,
    print_iterations=None,
    checkpoint_iterations=None,
    use_saliency=False,
    saliency_mask=None,
    saliency_weight=1.0,
    use_adain=False,
    use_patch_style=False,
    patch_weight=1.0,
    use_regions=False,
    region_masks=None,
    region_style_weights=None,
):
    """
    Stylize images.

    This function yields tuples (iteration, image, loss_vals) at every
    iteration. However `image` and `loss_vals` are None by default. Each
    `checkpoint_iterations`, `image` is not None. Each `print_iterations`,
    `loss_vals` is not None.

    `loss_vals` is a dict with loss values for the current iteration, e.g.
    ``{'content': 1.23, 'style': 4.56, 'tv': 7.89, 'total': 13.68}``.

    :rtype: iterator[tuple[int,image]]
    """
    shape = (1,) + content.shape
    style_shapes = [(1,) + style.shape for style in styles]
    content_features = {}
    style_features = [{} for _ in styles]
    style_stats = [{} for _ in styles]
    style_patches = [{} for _ in styles]

    vgg_weights, vgg_mean_pixel = vgg.load_net(network)

    layer_weight = 1.0
    style_layers_weights = {}
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] = layer_weight
        layer_weight *= style_layer_weight_exp

    # normalize style layer weights
    layer_weights_sum = 0
    for style_layer in STYLE_LAYERS:
        layer_weights_sum += style_layers_weights[style_layer]
    for style_layer in STYLE_LAYERS:
        style_layers_weights[style_layer] /= layer_weights_sum

    # compute content features in feedforward mode
    g = tf.Graph()
    with g.as_default(), g.device("/cpu:0"), tf.compat.v1.Session() as sess:
        image = tf.compat.v1.placeholder("float", shape=shape)
        net = vgg.net_preloaded(vgg_weights, image, pooling)
        content_pre = np.array([vgg.preprocess(content, vgg_mean_pixel)])
        for layer in CONTENT_LAYERS:
            content_features[layer] = net[layer].eval(feed_dict={image: content_pre})

    # compute style features in feedforward mode
    for i in range(len(styles)):
        g = tf.Graph()
        with g.as_default(), g.device("/cpu:0"), tf.compat.v1.Session() as sess:
            image = tf.compat.v1.placeholder("float", shape=style_shapes[i])
            net = vgg.net_preloaded(vgg_weights, image, pooling)
            style_pre = np.array([vgg.preprocess(styles[i], vgg_mean_pixel)])
            for layer in STYLE_LAYERS:
                features = net[layer].eval(feed_dict={image: style_pre})
                # Gram targets (default style loss)
                feats2 = np.reshape(features, (-1, features.shape[3]))
                gram = np.matmul(feats2.T, feats2) / feats2.size
                style_features[i][layer] = gram

                # AdaIN stats targets (mean/std per channel)
                mu = feats2.mean(axis=0)
                sigma = feats2.std(axis=0) + 1e-8
                style_stats[i][layer] = (mu.astype(np.float32), sigma.astype(np.float32))

            # Patch targets (only for a single mid-level layer to keep it lightweight)
            # Stored as normalized patch vectors for cosine-sim matching.
            if use_patch_style:
                patch_layer = "relu3_1"
                if patch_layer in net:
                    feats = net[patch_layer].eval(feed_dict={image: style_pre})  # (1,h,w,c)
                    style_patches[i][patch_layer] = _extract_style_patches_numpy(feats, patch_size=3, stride=1)

    initial_content_noise_coeff = 1.0 - initial_noiseblend

    # make stylized image using backpropogation
    with tf.Graph().as_default():
        if initial is None:
            initial = tf.random.normal(shape) * 0.256
        else:
            initial = np.array([vgg.preprocess(initial, vgg_mean_pixel)])
            initial = initial.astype("float32")
            initial = (initial) * initial_content_noise_coeff + (tf.random.normal(shape) * 0.256) * (
                1.0 - initial_content_noise_coeff
            )
        image = tf.Variable(initial)
        net = vgg.net_preloaded(vgg_weights, image, pooling)

        content_pre_const = tf.constant(np.array([vgg.preprocess(content, vgg_mean_pixel)]).astype(np.float32))

        saliency_ph = None
        if use_saliency:
            if saliency_mask is None:
                raise ValueError("use_saliency=True requires saliency_mask")
            sm = saliency_mask.astype(np.float32)
            if sm.shape != content.shape[:2]:
                raise ValueError("saliency_mask must match content spatial dimensions (H,W)")
            saliency_ph = tf.constant(sm[None, :, :, None], dtype=tf.float32)  # (1,H,W,1)

        region_ph = None
        if use_regions:
            if region_masks is None:
                raise ValueError("use_regions=True requires region_masks")
            if region_masks.shape[:2] != content.shape[:2]:
                raise ValueError("region_masks must match content spatial dimensions (H,W,K)")
            if region_style_weights is None:
                region_style_weights = [1.0 / float(region_masks.shape[2]) for _ in range(region_masks.shape[2])]
            if len(region_style_weights) != region_masks.shape[2]:
                raise ValueError("region_style_weights must have length K (matching region_masks.shape[2])")
            region_ph = tf.constant(region_masks[None, :, :, :].astype(np.float32), dtype=tf.float32)  # (1,H,W,K)

        # content loss
        content_layers_weights = {}
        content_layers_weights["relu4_2"] = content_weight_blend
        content_layers_weights["relu5_2"] = 1.0 - content_weight_blend

        content_loss = 0
        content_losses = []
        for content_layer in CONTENT_LAYERS:
            content_losses.append(
                content_layers_weights[content_layer]
                * content_weight
                * (
                    2
                    * tf.nn.l2_loss(net[content_layer] - content_features[content_layer])
                    / content_features[content_layer].size
                )
            )
        content_loss += reduce(tf.add, content_losses)

        # style loss (Gram default; optionally AdaIN; optionally saliency/regions weighting)
        style_loss = tf.constant(0.0, dtype=tf.float32)
        for i in range(len(styles)):
            style_losses = []
            for style_layer in STYLE_LAYERS:
                layer = net[style_layer]  # (1,h,w,c)

                spatial_mask = None
                if saliency_ph is not None:
                    # Reduce style in salient regions: use (1 - saliency)
                    spatial_mask = 1.0 - _resize_mask_to_layer(saliency_ph, layer)

                if region_ph is not None:
                    # Region-aware weighting: compute per-region style loss and weighted sum
                    region_losses = []
                    k = int(region_masks.shape[2])
                    for r in range(k):
                        region_mask = region_ph[:, :, :, r : r + 1]
                        region_mask = _resize_mask_to_layer(region_mask, layer)
                        combined = region_mask if spatial_mask is None else (region_mask * spatial_mask)
                        region_losses.append(
                            float(region_style_weights[r])
                            * _style_layer_loss(
                                layer=layer,
                                style_target_gram=style_features[i][style_layer],
                                style_target_stats=style_stats[i][style_layer],
                                use_adain=use_adain,
                                mask=combined,
                            )
                        )
                    style_losses.append(style_layers_weights[style_layer] * reduce(tf.add, region_losses))
                else:
                    style_losses.append(
                        style_layers_weights[style_layer]
                        * _style_layer_loss(
                            layer=layer,
                            style_target_gram=style_features[i][style_layer],
                            style_target_stats=style_stats[i][style_layer],
                            use_adain=use_adain,
                            mask=spatial_mask,
                        )
                    )

            style_loss += style_weight * style_blend_weights[i] * reduce(tf.add, style_losses)

        # optional patch-based style loss (lightweight, mid-level features)
        patch_loss = tf.constant(0.0, dtype=tf.float32)
        if use_patch_style:
            patch_layer = "relu3_1"
            if patch_layer in net and any(patch_layer in style_patches[i] for i in range(len(styles))):
                layer = net[patch_layer]
                spatial_mask = None
                if saliency_ph is not None:
                    spatial_mask = 1.0 - _resize_mask_to_layer(saliency_ph, layer)
                patch_terms = []
                for i in range(len(styles)):
                    if patch_layer not in style_patches[i]:
                        continue
                    sp = tf.constant(style_patches[i][patch_layer], dtype=tf.float32)  # (Ns, D)
                    patch_terms.append(_patch_style_loss(layer, sp, mask=spatial_mask))
                if patch_terms:
                    patch_loss = patch_weight * reduce(tf.add, patch_terms) / float(len(patch_terms))

        # total variation denoising
        tv_y_size = _tensor_size(image[:, 1:, :, :])
        tv_x_size = _tensor_size(image[:, :, 1:, :])
        tv_loss = (
            tv_weight
            * 2
            * (
                (tf.nn.l2_loss(image[:, 1:, :, :] - image[:, : shape[1] - 1, :, :]) / tv_y_size)
                + (tf.nn.l2_loss(image[:, :, 1:, :] - image[:, :, : shape[2] - 1, :]) / tv_x_size)
            )
        )

        # optional saliency preservation loss (pixel-level, salient areas)
        saliency_loss = tf.constant(0.0, dtype=tf.float32)
        if saliency_ph is not None:
            diff = (image - content_pre_const) * saliency_ph
            denom = tf.reduce_sum(saliency_ph) * 3.0 + 1e-8
            saliency_loss = saliency_weight * (2.0 * tf.nn.l2_loss(diff) / denom)

        # total loss
        loss = content_loss + style_loss + tv_loss + saliency_loss + patch_loss

        # We use OrderedDict to make sure we have the same order of loss types
        # (content, tv, style, total) as defined by the initial costruction of
        # the loss_store dict. This is important for print_progress() and
        # saving loss_arrs (column order) in the main script.
        #
        # Subtle Gotcha (tested with Python 3.5): The syntax
        # OrderedDict(key1=val1, key2=val2, ...) does /not/ create the same
        # order since, apparently, it first creates a normal dict with random
        # order (< Python 3.7) and then wraps that in an OrderedDict. We have
        # to pass in a data structure which is already ordered. I'd call this a
        # bug, since both constructor syntax variants result in different
        # objects. In 3.6, the order is preserved in dict() in CPython, in 3.7
        # they finally made it part of the language spec. Thank you!
        loss_store = OrderedDict(
            [
                ("content", content_loss),
                ("style", style_loss),
                ("tv", tv_loss),
                ("saliency", saliency_loss),
                ("patch", patch_loss),
                ("total", loss),
            ]
        )

        # optimizer setup
        train_step = tf.compat.v1.train.AdamOptimizer(learning_rate, beta1, beta2, epsilon).minimize(loss)

        # optimization
        best_loss = float("inf")
        best = None
        with tf.compat.v1.Session() as sess:
            sess.run(tf.compat.v1.global_variables_initializer())
            print("Optimization started...")
            if print_iterations and print_iterations != 0:
                print_progress(get_loss_vals(loss_store))
            iteration_times = []
            start = time.time()
            for i in range(iterations):
                iteration_start = time.time()
                if i > 0:
                    elapsed = time.time() - start
                    # take average of last couple steps to get time per iteration
                    remaining = np.mean(iteration_times[-10:]) * (iterations - i)
                    print(f"Iteration {i + 1:4d}/{iterations:4d} ({hms(elapsed)} elapsed, {hms(remaining)} remaining)")
                else:
                    print(f"Iteration {i + 1:4d}/{iterations:4d}")
                train_step.run()

                last_step = i == iterations - 1
                if last_step or (print_iterations and i % print_iterations == 0):
                    loss_vals = get_loss_vals(loss_store)
                    print_progress(loss_vals)
                else:
                    loss_vals = None

                if (checkpoint_iterations and i % checkpoint_iterations == 0) or last_step:
                    this_loss = loss.eval()
                    if this_loss < best_loss:
                        best_loss = this_loss
                        best = image.eval()

                    img_out = vgg.unprocess(best.reshape(shape[1:]), vgg_mean_pixel)

                    if preserve_colors:
                        original_image = np.clip(content, 0, 255)
                        styled_image = np.clip(img_out, 0, 255)

                        # Luminosity transfer steps:
                        # 1. Convert stylized RGB->grayscale accoriding to Rec.601 luma (0.299, 0.587, 0.114)
                        # 2. Convert stylized grayscale into YUV (YCbCr)
                        # 3. Convert original image into YUV (YCbCr)
                        # 4. Recombine (stylizedYUV.Y, originalYUV.U, originalYUV.V)
                        # 5. Convert recombined image from YUV back to RGB

                        # 1
                        styled_grayscale = rgb2gray(styled_image)
                        styled_grayscale_rgb = gray2rgb(styled_grayscale)

                        # 2
                        styled_grayscale_yuv = np.array(
                            Image.fromarray(styled_grayscale_rgb.astype(np.uint8)).convert("YCbCr")
                        )

                        # 3
                        original_yuv = np.array(Image.fromarray(original_image.astype(np.uint8)).convert("YCbCr"))

                        # 4
                        w, h, _ = original_image.shape
                        combined_yuv = np.empty((w, h, 3), dtype=np.uint8)
                        combined_yuv[..., 0] = styled_grayscale_yuv[..., 0]
                        combined_yuv[..., 1] = original_yuv[..., 1]
                        combined_yuv[..., 2] = original_yuv[..., 2]

                        # 5
                        img_out = np.array(Image.fromarray(combined_yuv, "YCbCr").convert("RGB"))
                else:
                    img_out = None

                yield i + 1 if last_step else i, img_out, loss_vals

                iteration_end = time.time()
                iteration_times.append(iteration_end - iteration_start)


def _tensor_size(tensor):
    from operator import mul

    return reduce(mul, (d.value for d in tensor.get_shape()), 1)


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.299, 0.587, 0.114])


def gray2rgb(gray):
    w, h = gray.shape
    rgb = np.empty((w, h, 3), dtype=np.float32)
    rgb[:, :, 2] = rgb[:, :, 1] = rgb[:, :, 0] = gray
    return rgb


def hms(seconds):
    seconds = int(seconds)
    hours = seconds // (60 * 60)
    minutes = (seconds // 60) % 60
    seconds = seconds % 60
    if hours > 0:
        return f"{hours:d} hr {minutes:d} min"
    elif minutes > 0:
        return f"{minutes:d} min {seconds:d} sec"
    else:
        return f"{seconds:d} sec"
