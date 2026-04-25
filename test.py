from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
from skimage.metrics import structural_similarity as ssim
import numpy as np

from models.adain_model import StyleTransferNet
from losses.losses import StyleAccuracyMetric


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.cpu() * STD.cpu() + MEAN.cpu()


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


class Evaluator:
    def __init__(self, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        self.vgg = nn.Sequential(
            *list(torch.hub.load('pytorch/vision', 'vgg19', pretrained=True).features.children())
        ).to(device)
        set_requires_grad(self.vgg, False)
        self.style_metric = StyleAccuracyMetric().to(device)

    def compute_content_similarity(
        self,
        content_img: np.ndarray,
        output_img: np.ndarray,
    ) -> float:
        if content_img.shape != output_img.shape:
            output_img = np.array(
                Image.fromarray((output_img * 255).astype(np.uint8)).resize(
                    (content_img.shape[1], content_img.shape[0])
                )
            )

        return ssim(
            content_img,
            output_img,
            data_range=1.0,
            channel_axis=2,
            win_size=11,
        )

    def compute_style_score(
        self,
        style_tensor: torch.Tensor,
        output_tensor: torch.Tensor,
    ) -> float:
        style_feat = self.vgg(style_tensor)
        output_feat = self.vgg(output_tensor)

        similarity = self.style_metric.style_similarity_score(style_feat, output_feat)
        return similarity.item()

    def evaluate(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        output: torch.Tensor,
    ) -> dict[str, float]:
        style_score = self.compute_style_score(style, output)

        content_np = content.squeeze().permute(1, 2, 0).cpu().detach().numpy()
        output_np = output.squeeze().permute(1, 2, 0).cpu().detach().numpy()

        content_ssim = self.compute_content_similarity(content_np, output_np)

        return {
            'style_similarity': style_score,
            'content_ssim': content_ssim,
        }


def load_and_preprocess_image(
    image_path: str,
    size: int = 512,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> torch.Tensor:
    transform = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)
    return tensor


def tensor_to_image(tensor: torch.Tensor) -> np.ndarray:
    tensor = denormalize(tensor)
    tensor = tensor.squeeze().permute(1, 2, 0)
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.cpu().detach().numpy()


def save_output(tensor: torch.Tensor, output_path: str) -> None:
    img = tensor_to_image(tensor)
    img = (img * 255).astype(np.uint8)
    Image.fromarray(img).save(output_path, quality=95)
    print(f"Output saved: {output_path}")


def test(
    model_path: str,
    content_image: str,
    style_image: str,
    output_path: str,
    image_size: int = 512,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    compute_metrics: bool = True,
) -> dict[str, float]:
    print(f"Loading model from: {model_path}")
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)

    net = StyleTransferNet().to(device)
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()

    print(f"Loading content image: {content_image}")
    content = load_and_preprocess_image(content_image, size=image_size, device=device)

    print(f"Loading style image: {style_image}")
    style = load_and_preprocess_image(style_image, size=image_size, device=device)

    with torch.no_grad():
        output = net(content, style)

    os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
    save_output(output, output_path)

    metrics = {}
    if compute_metrics:
        print("\nComputing evaluation metrics...")
        evaluator = Evaluator(device=device)

        metrics = evaluator.evaluate(content, style, output)

        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)
        print(f"Style Similarity Score: {metrics['style_similarity']:.4f}")
        print(f"Content SSIM Score:     {metrics['content_ssim']:.4f}")
        print("=" * 50)

        results_path = output_path.rsplit('.', 1)[0] + '_metrics.txt'
        with open(results_path, 'w') as f:
            f.write(f"Model: {model_path}\n")
            f.write(f"Content: {content_image}\n")
            f.write(f"Style: {style_image}\n")
            f.write(f"\nMetrics:\n")
            f.write(f"  Style Similarity Score: {metrics['style_similarity']:.4f}\n")
            f.write(f"  Content SSIM Score:     {metrics['content_ssim']:.4f}\n")
        print(f"Metrics saved: {results_path}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer Testing')
    parser.add_argument('--model', type=str, required=True, help='Path to trained model checkpoint')
    parser.add_argument('--content', type=str, required=True, help='Path to content image')
    parser.add_argument('--style', type=str, required=True, help='Path to style image')
    parser.add_argument('--output', type=str, required=True, help='Path to save output image')
    parser.add_argument('--image-size', type=int, default=512, help='Image size for inference')
    parser.add_argument('--no-metrics', action='store_true', help='Skip metrics computation')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    test(
        model_path=args.model,
        content_image=args.content,
        style_image=args.style,
        output_path=args.output,
        image_size=args.image_size,
        device=device,
        compute_metrics=not args.no_metrics,
    )


if __name__ == '__main__':
    main()