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

        self.content_dirs = [Path(d) for d in content_dir]
        self.style_dirs = [Path(d) for d in style_dir]

        self.content_images = []
        for d in self.content_dirs:
            self.content_images.extend(list(d.glob('*.jpg')))
            self.content_images.extend(list(d.glob('*.jpeg')))
            self.content_images.extend(list(d.glob('*.png')))

        self.style_images = []
        for d in self.style_dirs:
            self.style_images.extend(list(d.glob('*.jpg')))
            self.style_images.extend(list(d.glob('*.jpeg')))
            self.style_images.extend(list(d.glob('*.png')))

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