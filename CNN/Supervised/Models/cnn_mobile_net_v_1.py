"""
MobileNet V1 (CIFAR) — single-model supervised.

Depthwise separable convolutions with width multiplier α (Howard et al., 2017).
CIFAR-friendly tweak: initial 3×3 stem uses stride=1 (not 2). Remaining strides follow V1.
Output uses GAP → FC (no logits BN). Activation is ReLU6; BN after each conv.

Config mirrors the common V1 layout (with H×W downsampled to 2×2 for 32×32 inputs):
    conv3x3 s1 → [dw s1, pw 64] → [dw s2, pw 128] → [dw s1, pw 128]
    → [dw s2, pw 256] → [dw s1, pw 256] → [dw s2, pw 512] → 5×[dw s1, pw 512]
    → [dw s2, pw 1024] → [dw s1, pw 1024] → GAP → FC

Matches your modular factory style and includes a param_count helper.
"""
from __future__ import annotations
from typing import List
import math
import torch
import torch.nn as nn

__all__ = ["MobileNetV1CIFAR", "make_mobilenet_v1_cifar", "DepthwiseSeparable", "ConvBNReLU6"]


def _make_divisible(v: int | float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int = 0):
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

class DepthwiseSeparable(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int):
        super().__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU6(inplace=True),
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.dw(x)
        x = self.pw(x)
        return x

class MobileNetV1CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0, dropout: float = 0.0):
        super().__init__()
        a = width_mult
        def c(ch):
            return _make_divisible(ch * a)

        # Stem (stride=1 for CIFAR)
        self.stem = ConvBNReLU6(in_channels, c(32), k=3, s=1, p=1)

        layers: List[nn.Module] = []
        # Sequence per MobileNet V1
        cfg = [
            # (out_ch, stride)
            (64, 1),
            (128, 2), (128, 1),
            (256, 2), (256, 1),
            (512, 2),
            (512, 1), (512, 1), (512, 1), (512, 1), (512, 1),
            (1024, 2), (1024, 1),
        ]
        in_ch = c(32)
        for out_ch, s in cfg:
            layers.append(DepthwiseSeparable(in_ch, c(out_ch), stride=s))
            in_ch = c(out_ch)
        self.features = nn.Sequential(*layers)

        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.fc = nn.Linear(in_ch, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_mobilenet_v1_cifar(num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0, dropout: float = 0.0) -> MobileNetV1CIFAR:
    return MobileNetV1CIFAR(num_classes=num_classes, in_channels=in_channels, width_mult=width_mult, dropout=dropout)
