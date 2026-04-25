from __future__ import annotations

import torch
import torch.nn as nn


class AdaIN(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, content_feat: torch.Tensor, style_feat: torch.Tensor) -> torch.Tensor:
        content_mean = content_feat.mean(dim=[2, 3], keepdim=True)
        content_std = content_feat.std(dim=[2, 3], keepdim=True) + 1e-6
        style_mean = style_feat.mean(dim=[2, 3], keepdim=True)
        style_std = style_feat.std(dim=[2, 3], keepdim=True) + 1e-6

        normalized = (content_feat - content_mean) / content_std
        return normalized * style_std + style_mean


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.InstanceNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.InstanceNorm2d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Conv2d(512, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(256),
            nn.Conv2d(256, 256, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(256),
            nn.Conv2d(256, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(128),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(128),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='nearest'),
            ResidualBlock(64),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, 3, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(x)


class StyleTransferNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = self._build_encoder()
        self.adain = AdaIN()
        self.decoder = Decoder()

    def _build_encoder(self) -> nn.Module:
        import torchvision.models as models
        vgg = models.vgg19(weights='IMAGENET1K_V1').features
        encoder = nn.Sequential(*list(vgg.children()))
        return encoder

    def extract_features(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        content_feat = self.extract_features(content)
        style_feat = self.extract_features(style)

        stylized_feat = self.adain(content_feat, style_feat)
        output = self.decoder(stylized_feat)
        return output

    def generate(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        return self.forward(content, style)