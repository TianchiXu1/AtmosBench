"""Model architectures used in the AtmosBench evaluation."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


def _group_count(channels: int) -> int:
    for groups in (8, 6, 4, 3, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *, dropout: float = 0.0) -> None:
        super().__init__()
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.SiLU(),
        ]
        if dropout > 0:
            layers.append(nn.Dropout2d(dropout))
        layers.extend(
            [
                nn.Conv2d(out_channels, out_channels, 3, padding=1),
                nn.GroupNorm(_group_count(out_channels), out_channels),
                nn.SiLU(),
            ]
        )
        self.layers = nn.Sequential(*layers)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.layers(inputs)


class LogUNetBaseline(nn.Module):
    def __init__(self, *, in_channels: int, base_channels: int, depth: int, dropout: float = 0.0, out_channels: int = 1) -> None:
        super().__init__()
        if depth != 3:
            raise ValueError("registered LogUNet-M supports depth=3")
        c1, c2, c3, c4 = base_channels, base_channels * 2, base_channels * 4, base_channels * 8
        self.enc1 = DoubleConv(in_channels, c1, dropout=dropout)
        self.enc2 = DoubleConv(c1, c2, dropout=dropout)
        self.enc3 = DoubleConv(c2, c3, dropout=dropout)
        self.bottleneck = DoubleConv(c3, c4, dropout=dropout)
        self.dec3 = DoubleConv(c4 + c3, c3, dropout=dropout)
        self.dec2 = DoubleConv(c3 + c2, c2, dropout=dropout)
        self.dec1 = DoubleConv(c2 + c1, c1, dropout=dropout)
        self.head = nn.Conv2d(c1, out_channels, 3, padding=1)

    @staticmethod
    def _up_to(inputs: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        return F.interpolate(inputs, size=reference.shape[-2:], mode="bilinear", align_corners=False)

    def forward(self, inputs: torch.Tensor) -> dict[str, torch.Tensor]:
        enc1 = self.enc1(inputs)
        enc2 = self.enc2(F.avg_pool2d(enc1, 2))
        enc3 = self.enc3(F.avg_pool2d(enc2, 2))
        bottleneck = self.bottleneck(F.avg_pool2d(enc3, 2))
        dec3 = self.dec3(torch.cat([self._up_to(bottleneck, enc3), enc3], dim=1))
        dec2 = self.dec2(torch.cat([self._up_to(dec3, enc2), enc2], dim=1))
        dec1 = self.dec1(torch.cat([self._up_to(dec2, enc1), enc1], dim=1))
        raw = self.head(dec1)
        outputs = {"mean_norm": torch.sigmoid(raw[:, 0:1])}
        if raw.shape[1] > 1:
            outputs["logvar_norm"] = raw[:, 1:2].clamp(-12.0, 4.0)
        return outputs
