from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ContentLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.criterion = nn.MSELoss()

    def forward(self, content_feat: torch.Tensor, output_feat: torch.Tensor) -> torch.Tensor:
        return self.criterion(content_feat, output_feat)


class StyleLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def gram_matrix(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.size()
        features = feat.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (c * h * w)
        return gram

    def forward(self, style_feat: torch.Tensor, output_feat: torch.Tensor) -> torch.Tensor:
        style_gram = self.gram_matrix(style_feat)
        output_gram = self.gram_matrix(output_feat)
        return F.mse_loss(output_gram, style_gram)


class TotalVariationLoss(nn.Module):
    def __init__(self, weight: float = 1e-5):
        super().__init__()
        self.weight = weight

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, c, h, w = x.size()
        tv_h = torch.abs(x[:, :, 1:, :] - x[:, :, :-1, :]).sum()
        tv_w = torch.abs(x[:, :, :, 1:] - x[:, :, :, :-1]).sum()
        return self.weight * (tv_h + tv_w) / (batch_size * c * h * w)


class PerceptualLoss(nn.Module):
    def __init__(
        self,
        content_weight: float = 1.0,
        style_weight: float = 1.0,
        tv_weight: float = 1e-5,
    ):
        super().__init__()
        self.content_weight = content_weight
        self.style_weight = style_weight
        self.tv_weight = tv_weight

        self.content_loss = ContentLoss()
        self.style_loss = StyleLoss()
        self.tv_loss = TotalVariationLoss(weight=tv_weight)

    def forward(
        self,
        content_feat: torch.Tensor,
        style_feat: torch.Tensor,
        output: torch.Tensor,
        content_output: torch.Tensor,
        style_output: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        content = self.content_weight * self.content_loss(content_output, content_feat)
        style = self.style_weight * self.style_loss(style_output, style_feat)
        tv = self.tv_weight * self.tv_loss(output)

        total = content + style + tv

        return {
            'total': total,
            'content': content,
            'style': style,
            'tv': tv,
        }


class StyleAccuracyMetric(nn.Module):
    def __init__(self):
        super().__init__()

    def gram_matrix(self, feat: torch.Tensor) -> torch.Tensor:
        b, c, h, w = feat.size()
        features = feat.view(b, c, h * w)
        gram = torch.bmm(features, features.transpose(1, 2))
        gram = gram / (c * h * w)
        return gram

    def style_similarity_score(
        self,
        style_feat: torch.Tensor,
        output_feat: torch.Tensor,
    ) -> torch.Tensor:
        style_gram = self.gram_matrix(style_feat)
        output_gram = self.gram_matrix(output_feat)
        similarity = 1.0 - F.mse_loss(output_gram, style_gram)
        return similarity.clamp(0, 1)

    def forward(
        self,
        output: torch.Tensor,
        content: torch.Tensor,
        style: torch.Tensor,
    ) -> dict[str, float]:
        return {
            'style_similarity': self.style_similarity_score(style, output).item(),
        }