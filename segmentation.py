from __future__ import annotations

import numpy as np


def _normalize_01(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32)
    x = x - float(x.min())
    return x / (float(x.max()) + eps)


def _kmeans(pixels: np.ndarray, k: int, iters: int = 15, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """
    Very small k-means implementation for color clustering.

    pixels: (N, 3) float32 in [0, 1]
    Returns (centers (k,3), labels (N,))
    """
    rng = np.random.default_rng(seed)
    n = pixels.shape[0]
    if n == 0:
        raise ValueError("No pixels to cluster")

    # Init: pick random unique points
    idx = rng.choice(n, size=min(k, n), replace=False)
    centers = pixels[idx].copy()
    if centers.shape[0] < k:
        # pad by repeating if image is tiny
        reps = k - centers.shape[0]
        centers = np.concatenate([centers, centers[:reps]], axis=0)

    labels = np.zeros((n,), dtype=np.int32)
    for _ in range(iters):
        # Assign
        d2 = ((pixels[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)  # (N, k)
        new_labels = np.argmin(d2, axis=1).astype(np.int32)
        if np.array_equal(new_labels, labels):
            break
        labels = new_labels

        # Update
        for j in range(k):
            mask = labels == j
            if not np.any(mask):
                centers[j] = pixels[rng.integers(0, n)]
            else:
                centers[j] = pixels[mask].mean(axis=0)

    return centers.astype(np.float32), labels


def get_regions(image: np.ndarray, k: int = 3, downsample_max_side: int = 256) -> np.ndarray:
    """
    Region-aware masks via simple color clustering (k-means).

    Returns float32 masks shaped (H, W, K) with values in {0,1}.
    Masks are computed on a downsampled image for speed, then nearest-upsampled.
    """
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("get_regions expects an RGB image of shape (H, W, 3)")

    h, w, _ = image.shape
    scale = min(1.0, float(downsample_max_side) / float(max(h, w)))
    if scale < 1.0:
        nh = max(1, int(round(h * scale)))
        nw = max(1, int(round(w * scale)))
        # nearest downsample without extra deps
        ys = (np.linspace(0, h - 1, nh)).astype(np.int32)
        xs = (np.linspace(0, w - 1, nw)).astype(np.int32)
        small = image[ys][:, xs]
    else:
        nh, nw = h, w
        small = image

    pixels = (small.astype(np.float32) / 255.0).reshape(-1, 3)
    _, labels = _kmeans(pixels, k=k, iters=20, seed=0)
    labels2d = labels.reshape(nh, nw)

    masks_small = np.stack([(labels2d == i).astype(np.float32) for i in range(k)], axis=2)  # (nh,nw,k)

    if (nh, nw) != (h, w):
        # nearest upsample back to (h,w)
        ys = (np.linspace(0, nh - 1, h)).astype(np.int32)
        xs = (np.linspace(0, nw - 1, w)).astype(np.int32)
        masks = masks_small[ys][:, xs]
    else:
        masks = masks_small

    # Ensure every pixel has exactly one region (numerical safety)
    masks = (masks == masks.max(axis=2, keepdims=True)).astype(np.float32)

    # Order regions by size (largest first) for stable weighting
    sizes = masks.reshape(-1, k).sum(axis=0)
    order = np.argsort(-sizes)
    masks = masks[:, :, order]

    return masks.astype(np.float32)


def regions_to_label_map(region_masks: np.ndarray) -> np.ndarray:
    """
    Convert (H,W,K) one-hot-ish masks into a label map (H,W) in [0..K-1].
    """
    if region_masks.ndim != 3:
        raise ValueError("region_masks must be (H, W, K)")
    return np.argmax(region_masks, axis=2).astype(np.int32)


def visualize_regions(region_masks: np.ndarray) -> np.ndarray:
    """
    Create a simple visualization image for region masks: pseudo-colored labels.
    Returns uint8 RGB image (H,W,3).
    """
    labels = regions_to_label_map(region_masks)
    k = region_masks.shape[2]
    # Fixed palette
    palette = np.array(
        [
            [230, 25, 75],
            [60, 180, 75],
            [255, 225, 25],
            [0, 130, 200],
            [245, 130, 48],
            [145, 30, 180],
            [70, 240, 240],
            [240, 50, 230],
        ],
        dtype=np.uint8,
    )
    colors = palette[labels % palette.shape[0]]
    return colors.astype(np.uint8)

