"""
MobileNet V2 (CIFAR) — single-model supervised.

Inverted Residuals + Linear Bottlenecks with width multiplier α (Sandler et al., 2018).
CIFAR tweak: initial 3×3 conv stride=1. Uses ReLU6, BN. Residual only when stride=1 and in==out.

Typical CIFAR config (following V2 layout):
  stem(32, s1) → [t=1,c=16,n=1,s=1] → [t=6,c=24,n=2,s=2]
  → [t=6,c=32,n=3,s=2] → [t=6,c=64,n=4,s=2] → [t=6,c=96,n=3,s=1]
  → [t=6,c=160,n=3,s=2] → [t=6,c=320,n=1,s=1] → last 1×1 conv(1280) → GAP → FC

Factory mirrors your style and exposes width_mult, dropout.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

__all__ = [
    "MobileNetV2CIFAR",
    "make_mobilenet_v2_cifar",
    "InvertedResidual",
    "ConvBNReLU6",
]


def _make_divisible(v: int | float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int | None = None, groups: int = 1):
        if p is None:
            p = (k - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        assert stride in [1, 2]
        hidden_dim = int(round(in_ch * expand_ratio))
        self.use_res_connect = (stride == 1 and in_ch == out_ch)

        layers: List[nn.Module] = []
        if expand_ratio != 1:
            # pointwise expansion
            layers.append(ConvBNReLU6(in_ch, hidden_dim, k=1, s=1, p=0))
        # depthwise
        layers.append(ConvBNReLU6(hidden_dim, hidden_dim, k=3, s=stride, groups=hidden_dim))
        # project (linear; no nonlinearity)
        layers.append(nn.Conv2d(hidden_dim, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_res_connect:
            return x + out
        else:
            return out

class MobileNetV2CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0, dropout: float = 0.0):
        super().__init__()
        self.width_mult = width_mult
        def c(ch):
            return _make_divisible(ch * width_mult)

        # CIFAR stem: stride 1
        input_channel = c(32)
        layers: List[nn.Module] = [ConvBNReLU6(in_channels, input_channel, k=3, s=1)]

        # (t, c, n, s)
        cfg: List[Tuple[int, int, int, int]] = [
            (1, 16, 1, 1),
            (6, 24, 2, 2),
            (6, 32, 3, 2),
            (6, 64, 4, 2),
            (6, 96, 3, 1),
            (6, 160, 3, 2),
            (6, 320, 1, 1),
        ]

        # building inverted residual blocks
        for t, c_out, n, s in cfg:
            out_channel = c(c_out)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(InvertedResidual(input_channel, out_channel, stride, expand_ratio=t))
                input_channel = out_channel

        # last conv
        last_channel = c(1280) if width_mult > 1.0 else 1280
        layers.append(ConvBNReLU6(input_channel, last_channel, k=1, s=1, p=0))
        self.features = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(last_channel, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_mobilenet_v2_cifar(num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0, dropout: float = 0.0) -> MobileNetV2CIFAR:
    return MobileNetV2CIFAR(num_classes=num_classes, in_channels=in_channels, width_mult=width_mult, dropout=dropout)
