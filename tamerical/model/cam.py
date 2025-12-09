"""Character-Aided Module (CAM) to mimic ICAL auxiliary head."""

from __future__ import annotations

import torch
from torch import nn


class CharacterAidedModule(nn.Module):
    """Simple CNN head that predicts per-character probabilities."""

    def __init__(self, in_channels: int, vocab_size: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(in_channels // 2, vocab_size)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        x = self.net(feat)
        x = torch.flatten(x, 1)
        return self.classifier(x)
