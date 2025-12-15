from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """
    Standard CIFAR-style basic residual block:
      conv3x3 -> BN -> ReLU -> conv3x3 -> BN, with optional 1x1 downsample.
    """

    expansion = 1

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.downsample = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.dropout(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = out + self.downsample(identity)
        out = self.relu(out)
        return out


@dataclass
class ADPResNetConfig:
    input_channels: int = 3
    num_classes: int = 10
    width: int = 16  # base channels for stage 1
    depth: int = 2   # blocks per stage
    dropout: float = 0.0


class ADPResNet(nn.Module):
    """
    Simple ResNet-style backbone for CIFAR with ADP-friendly width/depth:
      - width: base channels for stage1 (stages 2/3 use 2x and 4x).
      - depth: number of BasicBlocks per stage (3 stages total).

    This exposes `width` and `depth` attributes so ADP algorithms can expand
    either dimension while attempting to preserve weights.
    """

    def __init__(self, cfg: ADPResNetConfig) -> None:
        super().__init__()

        self.input_channels = cfg.input_channels
        self.num_classes = cfg.num_classes
        self.width = cfg.width
        self.depth = cfg.depth
        self.dropout = cfg.dropout

        c1 = cfg.width
        c2 = cfg.width * 2
        c3 = cfg.width * 4

        self.stem = nn.Sequential(
            nn.Conv2d(cfg.input_channels, c1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c1),
            nn.ReLU(inplace=True),
        )

        self.stage1 = self._make_stage(c1, c1, cfg.depth, stride=1, dropout=cfg.dropout)
        self.stage2 = self._make_stage(c1, c2, cfg.depth, stride=2, dropout=cfg.dropout)
        self.stage3 = self._make_stage(c2, c3, cfg.depth, stride=2, dropout=cfg.dropout)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(c3, cfg.num_classes)

    @staticmethod
    def _make_stage(
        in_channels: int,
        out_channels: int,
        blocks: int,
        stride: int,
        dropout: float,
    ) -> nn.Sequential:
        layers: List[nn.Module] = []
        layers.append(BasicBlock(in_channels, out_channels, stride=stride, dropout=dropout))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels, stride=1, dropout=dropout))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


def make_adp_resnet(
    input_channels: int = 3,
    num_classes: int = 10,
    width: int = 16,
    depth: int = 2,
    dropout: float = 0.0,
) -> ADPResNet:
    """
    Convenience factory used by ADP and STL runners.
    """
    cfg = ADPResNetConfig(
        input_channels=input_channels,
        num_classes=num_classes,
        width=width,
        depth=depth,
        dropout=dropout,
    )
    return ADPResNet(cfg)


def estimate_neurons(width: int, depth: int) -> int:
    """
    Simple scalar complexity proxy for ADP:
      neurons ~ width * (depth + 1)
    This mirrors the ConvNetSTL ADP metric enough for comparisons while
    remaining cheap to compute.
    """
    return int(width * (depth + 1))

