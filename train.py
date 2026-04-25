from __future__ import annotations

import argparse
import os
from pathlib import Path
import time
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from models.adain_model import StyleTransferNet
from losses.losses import PerceptualLoss
from datasets.datasets import PairedDataset


MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def set_requires_grad(model: nn.Module, requires_grad: bool) -> None:
    for param in model.parameters():
        param.requires_grad = requires_grad


def denormalize(tensor: torch.Tensor) -> torch.Tensor:
    return tensor * STD + MEAN


def train(
    content_dir: str,
    style_dir: str | list[str],
    output_dir: str,
    epochs: int = 2,
    batch_size: int = 8,
    lr: float = 1e-4,
    image_size: int = 256,
    content_weight: float = 1.0,
    style_weight: float = 1e2,
    tv_weight: float = 1e-5,
    log_interval: int = 100,
    checkpoint_interval: int = 500,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
) -> None:
    os.makedirs(output_dir, exist_ok=True)

    dataset = PairedDataset(content_dir, style_dir, image_size)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
    )

    net = StyleTransferNet().to(device)
    set_requires_grad(net.encoder, False)
    criterion = PerceptualLoss(
        content_weight=content_weight,
        style_weight=style_weight,
        tv_weight=tv_weight,
    ).to(device)

    optimizer = Adam(net.decoder.parameters(), lr=lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=len(dataloader) * epochs, eta_min=1e-6)

    print(f"Training on {len(dataset)} images | Epochs: {epochs} | Batch size: {batch_size}")
    print(f"Device: {device}")
    print("-" * 50)

    global_step = 0
    start_time = time.time()

    for epoch in range(epochs):
        net.train()

        for iteration, (content_imgs, style_imgs) in enumerate(dataloader):
            content_imgs = content_imgs.to(device)
            style_imgs = style_imgs.to(device)

            optimizer.zero_grad()

            content_feat = net.encoder(content_imgs)
            style_feat = net.encoder(style_imgs)

            stylized_feat = net.adain(content_feat, style_feat)
            stylized = net.decoder(stylized_feat)

            output_feat = net.encoder(stylized)

            losses = criterion(
                content_feat=content_feat,
                style_feat=style_feat,
                output=stylized,
                content_output=output_feat,
                style_output=output_feat,
            )

            losses['total'].backward()
            torch.nn.utils.clip_grad_norm_(net.decoder.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            if global_step % log_interval == 0:
                elapsed = time.time() - start_time
                print(
                    f"[Epoch {epoch+1}/{epochs}] "
                    f"[Step {global_step}] "
                    f"Loss: {losses['total'].item():.4f} | "
                    f"Content: {losses['content'].item():.4f} | "
                    f"Style: {losses['style'].item():.4f} | "
                    f"Time: {elapsed:.1f}s"
                )

            if global_step > 0 and global_step % checkpoint_interval == 0:
                checkpoint_path = os.path.join(output_dir, f'checkpoint_{global_step}.pth')
                torch.save({
                    'epoch': epoch,
                    'global_step': global_step,
                    'model_state_dict': net.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, checkpoint_path)
                print(f"Checkpoint saved: {checkpoint_path}")

            global_step += 1

        print(f"Completed epoch {epoch+1}/{epochs}")

    final_path = os.path.join(output_dir, 'final_model.pth')
    torch.save({
        'epoch': epochs,
        'global_step': global_step,
        'model_state_dict': net.state_dict(),
    }, final_path)
    print(f"Model saved: {final_path}")


def main():
    parser = argparse.ArgumentParser(description='AdaIN Style Transfer Training')
    parser.add_argument('--content-dir', type=str, required=True, help='Path to content images directory')
    parser.add_argument('--style-dir', type=str, required=True, help='Path to style images directory')
    parser.add_argument('--output-dir', type=str, default='./checkpoints', help='Output directory for checkpoints')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--image-size', type=int, default=256, help='Image size for training')
    parser.add_argument('--content-weight', type=float, default=1.0, help='Content loss weight')
    parser.add_argument('--style-weight', type=float, default=1e2, help='Style loss weight')
    parser.add_argument('--tv-weight', type=float, default=1e-5, help='Total variation loss weight')
    parser.add_argument('--log-interval', type=int, default=100, help='Log interval')
    parser.add_argument('--checkpoint-interval', type=int, default=500, help='Checkpoint interval')
    parser.add_argument('--device', type=str, default=None, help='Device (cuda/cpu)')

    args = parser.parse_args()

    if args.device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device

    train(
        content_dir=args.content_dir,
        style_dir=args.style_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        image_size=args.image_size,
        content_weight=args.content_weight,
        style_weight=args.style_weight,
        tv_weight=args.tv_weight,
        log_interval=args.log_interval,
        checkpoint_interval=args.checkpoint_interval,
        device=device,
    )


if __name__ == '__main__':
    main()