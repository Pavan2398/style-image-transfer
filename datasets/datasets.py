from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']


def find_images(folder: str) -> list[Path]:
    folder_path = Path(folder)
    if not folder_path.exists():
        return []
    images = []
    for ext in IMAGE_EXTENSIONS:
        images.extend(list(folder_path.glob(f'*{ext}')))
        images.extend(list(folder_path.glob(f'*/*{ext}')))
    return images


class PairedDataset(Dataset):
    def __init__(
        self,
        content_dir: str | list[str],
        style_dir: str | list[str],
        image_size: int = 256,
        transform: Optional[transforms.Compose] = None,
    ):
        if isinstance(content_dir, str):
            content_dir = [content_dir]
        if isinstance(style_dir, str):
            style_dir = [style_dir]

        self.content_images = []
        for d in content_dir:
            self.content_images.extend(find_images(d))

        self.style_images = []
        for d in style_dir:
            self.style_images.extend(find_images(d))

        print(f"Found {len(self.content_images)} content images, {len(self.style_images)} style images")

        if transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=MEAN, std=STD),
            ])
        else:
            self.transform = transform

    def __len__(self) -> int:
        return min(len(self.content_images), len(self.style_images))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        content_path = self.content_images[index % len(self.content_images)]
        style_path = self.style_images[index % len(self.style_images)]

        content_img = Image.open(content_path).convert('RGB')
        style_img = Image.open(style_path).convert('RGB')

        content_tensor = self.transform(content_img)
        style_tensor = self.transform(style_img)

        return content_tensor, style_tensor