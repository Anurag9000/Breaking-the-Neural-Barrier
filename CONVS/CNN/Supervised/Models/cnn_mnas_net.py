"""
MnasNet (CIFAR) — single-model supervised.

Implements MnasNet-A style backbone (Tan et al., 2019) built from **MBConv** blocks
(inverted residual + depthwise + linear projection) with optional **SE**. CIFAR tweak:
stem stride=1; downsampling via stride-2 MBConv at the start of stages → spatial sizes 32→16→8→4.

Activation: ReLU6. Residual when stride=1 and in==out. Final 1×1 conv → GAP → FC.
Exposes width multiplier (α) and dropout, following your factory/run-file conventions.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

__all__ = [
    "MnasNetCIFAR",
    "make_mnasnet_cifar",
]


def _make_divisible(v: int | float, divisor: int = 8, min_value: int | None = None) -> int:
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w

class ConvBNReLU6(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int | None = None, groups: int = 1):
        if p is None:
            p = (k - 1) // 2
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU6(inplace=True),
        )

class MBConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand: int, use_se: bool):
        super().__init__()
        assert stride in (1, 2)
        hidden = in_ch * expand
        self.use_res = (stride == 1 and in_ch == out_ch)
        layers: List[nn.Module] = []
        if expand != 1:
            layers.append(ConvBNReLU6(in_ch, hidden, k=1, s=1, p=0))
        # depthwise
        layers.append(ConvBNReLU6(hidden, hidden, k=3, s=stride, groups=hidden))
        # SE
        if use_se:
            layers.append(SEModule(hidden))
        # projection (linear)
        layers.append(nn.Conv2d(hidden, out_ch, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_ch))
        self.block = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.block(x)
        if self.use_res:
            y = x + y
        return y

class MnasNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0, dropout: float = 0.0):
        super().__init__()
        a = width_mult
        def c(ch: int) -> int:
            return _make_divisible(ch * a)

        # Stem (stride=1 for CIFAR)
        input_ch = c(32)
        self.stem = ConvBNReLU6(in_channels, input_ch, k=3, s=1)

        # (t=expand, c=out, n=repeats, s=stride, se=use_se) — MnasNet-A style (CIFAR strides)
        cfg: List[Tuple[int,int,int,int,bool]] = [
            (1,  16, 1, 1, False),
            (6,  24, 2, 2, False),
            (6,  40, 3, 2, True ),
            (6,  80, 4, 2, False),
            (6, 112, 2, 1, True ),
            (6, 160, 3, 2, True ),
        ]

        layers: List[nn.Module] = []
        in_ch = input_ch
        for t, c_out, n, s, se in cfg:
            out_ch = c(c_out)
            for i in range(n):
                stride = s if i == 0 else 1
                layers.append(MBConv(in_ch, out_ch, stride=stride, expand=t, use_se=se))
                in_ch = out_ch
        self.features = nn.Sequential(*layers)

        # Last 1×1 conv
        last_ch = c(1280)
        self.head = ConvBNReLU6(in_ch, last_ch, k=1, s=1, p=0)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(last_ch, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.features(x)
        x = self.head(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_mnasnet_cifar(num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0, dropout: float = 0.0) -> MnasNetCIFAR:
    return MnasNetCIFAR(num_classes=num_classes, in_channels=in_channels, width_mult=width_mult, dropout=dropout)
