"""
MobileNet V3 (CIFAR) — single-model supervised.

Inverted Residual + (optional) SE + nonlinearity {ReLU, h-swish} with width multiplier α.
Implements CIFAR-adapted **V3-Small** and **V3-Large** (Howard et al., 2019), using stride-1 stem.
Blocks follow (k, exp, out, SE, NL, s). SE uses sigmoid (h-sigmoid) gate. NL∈{ReLU, h-swish}.
Head: 1×1 conv → h-swish → GAP → dropout → FC.

Factory mirrors your style, exposes version, width_mult, dropout.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

__all__ = [
    "MobileNetV3CIFAR",
    "make_mobilenet_v3_cifar",
]

# ----------------------
# Activations
# ----------------------
class HSigmoid(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.relu6 = nn.ReLU6(inplace=inplace)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu6(x + 3) / 6

class HSwish(nn.Module):
    def __init__(self, inplace: bool = True):
        super().__init__()
        self.hsig = HSigmoid(inplace)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.hsig(x)

# ----------------------
# SE Module
# ----------------------
class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.gate = HSigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w

# ----------------------
# Inverted Residual Block (V3)
# ----------------------
class InvertedResidualV3(nn.Module):
    def __init__(self, in_ch: int, exp_ch: int, out_ch: int, kernel: int, stride: int,
                 use_se: bool, use_hs: bool):
        super().__init__()
        assert stride in (1, 2)
        self.use_res_connect = (stride == 1 and in_ch == out_ch)
        Act = HSwish if use_hs else nn.ReLU
        padding = (kernel - 1) // 2

        layers: List[nn.Module] = []
        # 1) expand (if needed)
        if exp_ch != in_ch:
            layers += [nn.Conv2d(in_ch, exp_ch, 1, bias=False), nn.BatchNorm2d(exp_ch), Act(inplace=True)]
        # 2) depthwise
        layers += [
            nn.Conv2d(exp_ch, exp_ch, kernel, stride=stride, padding=padding, groups=exp_ch, bias=False),
            nn.BatchNorm2d(exp_ch),
            Act(inplace=True)
        ]
        # 3) SE (optional)
        if use_se:
            layers += [SEModule(exp_ch)]
        # 4) project (linear)
        layers += [nn.Conv2d(exp_ch, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch)]

        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.use_res_connect:
            out = x + out
        return out

# ----------------------
# Network
# ----------------------
class MobileNetV3CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, version: str = "small",
                 width_mult: float = 1.0, dropout: float = 0.0):
        super().__init__()
        assert version in {"small", "large"}
        self.version = version

        def c(ch: int) -> int:
            # V3 uses 8-divisible channel alignment commonly
            return max(8, int(ch * width_mult + 4) // 8 * 8)

        ActHead = HSwish

        # CIFAR stem (stride=1)
        input_ch = c(16)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, input_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(input_ch),
            HSwish(inplace=True),
        )

        # Block specs (k, exp, out, SE, NL(h-swish?), s)
        if version == "large":
            cfg: List[Tuple[int,int,int,bool,bool,int]] = [
                (3, 16, 16, False, False, 1),
                (3, 64, 24, False, False, 2),
                (3, 72, 24, False, False, 1),
                (5, 72, 40, True,  False, 2),
                (5, 120,40, True,  False, 1),
                (5, 120,40, True,  False, 1),
                (3, 240,80, False, True,  2),
                (3, 200,80, False, True,  1),
                (3, 184,80, False, True,  1),
                (3, 184,80, False, True,  1),
                (3, 480,112, True,  True,  1),
                (3, 672,112, True,  True,  1),
                (5, 672,160, True,  True,  2),
                (5, 960,160, True,  True,  1),
                (5, 960,160, True,  True,  1),
            ]
            last_ch = c(960)
        else:  # small
            cfg = [
                (3, 16, 16, True,  False, 2),
                (3, 72, 24, False, False, 2),
                (3, 88, 24, False, False, 1),
                (5, 96, 40, True,  True,  2),
                (5, 240,40, True,  True,  1),
                (5, 240,40, True,  True,  1),
                (5, 120,48, True,  True,  1),
                (5, 144,48, True,  True,  1),
                (5, 288,96, True,  True,  2),
                (5, 576,96, True,  True,  1),
                (5, 576,96, True,  True,  1),
            ]
            last_ch = c(576)

        layers: List[nn.Module] = []
        in_ch = input_ch
        for k, exp, out, se, hs, s in cfg:
            layers.append(InvertedResidualV3(in_ch, c(exp), c(out), k, s, use_se=se, use_hs=hs))
            in_ch = c(out)
        self.blocks = nn.Sequential(*layers)

        # Head
        head_ch = c(last_ch)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            ActHead(inplace=True),
        )
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        self.classifier = nn.Linear(head_ch, num_classes)
        nn.init.kaiming_normal_(self.classifier.weight, nonlinearity='relu')
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_mobilenet_v3_cifar(num_classes: int = 10, in_channels: int = 3, version: str = "small",
                             width_mult: float = 1.0, dropout: float = 0.0) -> MobileNetV3CIFAR:
    return MobileNetV3CIFAR(num_classes=num_classes, in_channels=in_channels, version=version,
                            width_mult=width_mult, dropout=dropout)
