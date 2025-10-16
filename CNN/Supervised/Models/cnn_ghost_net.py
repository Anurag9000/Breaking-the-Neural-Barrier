"""
GhostNet (CIFAR) — single-model supervised.

Implements GhostNet (Han et al., 2020) with Ghost modules producing "cheap" feature maps via
depthwise operations. Uses Ghost Bottlenecks (GBNeck) with optional SE, stride∈{1,2}.
CIFAR tweaks: stem stride=1; stage-downsampling via stride-2 GBNecks to reach 4× spatial map.

Key pieces
  • GhostModule: out = concat(primary 1×1 conv, cheap DW conv on primary)[:out_ch]
  • GBNeck: Ghost(expansion) → (SE?) → DW (s) → Ghost(project linear); residual if stride=1 & in==out
  • Activations: ReLU early, H-Swish in later stages (as per paper); configurable width_mult & ghost_ratio

Factory mirrors your style and includes param_count helper.
"""
from __future__ import annotations
from typing import List, Tuple
import math
import torch
import torch.nn as nn

__all__ = [
    "GhostNetCIFAR",
    "make_ghostnet_cifar",
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
# Ghost module
# ----------------------
class GhostModule(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 1, stride: int = 1, relu: bool = True, ratio: int = 2):
        super().__init__()
        init_channels = math.ceil(out_ch / ratio)
        new_channels = init_channels * (ratio - 1)
        act = nn.ReLU(inplace=True) if relu else nn.Identity()
        padding = (kernel_size - 1) // 2
        self.primary = nn.Sequential(
            nn.Conv2d(in_ch, init_channels, kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(init_channels),
            act,
        )
        self.cheap = nn.Sequential(
            nn.Conv2d(init_channels, new_channels, kernel_size=3, stride=1, padding=1, groups=init_channels, bias=False),
            nn.BatchNorm2d(new_channels),
            act,
        )
        self.out_ch = out_ch

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.primary(x)
        x2 = self.cheap(x1)
        out = torch.cat([x1, x2], dim=1)
        return out[:, : self.out_ch, :, :]

# ----------------------
# Ghost Bottleneck
# ----------------------
class GhostBottleneck(nn.Module):
    def __init__(self, in_ch: int, hidden_ch: int, out_ch: int, kernel: int, stride: int, use_se: bool, act_hs: bool, ratio: int = 2):
        super().__init__()
        assert stride in (1, 2)
        Act = HSwish if act_hs else nn.ReLU
        pad = (kernel - 1) // 2
        # Pointwise Ghost (expand)
        self.ghost1 = GhostModule(in_ch, hidden_ch, kernel_size=1, stride=1, relu=True, ratio=ratio)
        # Depthwise conv for stride
        if stride == 2:
            self.dw = nn.Sequential(
                nn.Conv2d(hidden_ch, hidden_ch, kernel_size=kernel, stride=stride, padding=pad, groups=hidden_ch, bias=False),
                nn.BatchNorm2d(hidden_ch),
            )
        else:
            self.dw = nn.Identity()
        # SE (optional)
        self.se = SEModule(hidden_ch) if use_se else nn.Identity()
        # Project Ghost (linear)
        self.ghost2 = GhostModule(hidden_ch, out_ch, kernel_size=1, stride=1, relu=False, ratio=ratio)
        # Shortcut
        if stride == 1 and in_ch == out_ch:
            self.shortcut = nn.Identity()
        else:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, in_ch, kernel_size=kernel, stride=stride, padding=pad, groups=in_ch, bias=False),
                nn.BatchNorm2d(in_ch),
                nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        self.act = Act(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.ghost1(x)
        out = self.dw(out)
        out = self.se(out)
        out = self.ghost2(out)
        out = out + self.shortcut(x)
        out = self.act(out)
        return out

# ----------------------
# Network
# ----------------------
class GhostNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0, ghost_ratio: int = 2, dropout: float = 0.0):
        super().__init__()
        self.width_mult = width_mult
        self.ghost_ratio = ghost_ratio

        def c(ch: int) -> int:
            # align to multiple of 4 (common in GhostNet)
            return max(4, int(round(ch * width_mult / 4)) * 4)

        # CIFAR stem (stride=1)
        out_ch = c(16)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
        in_ch = out_ch

        # (k, exp, out, se, s, hs)
        cfg: List[Tuple[int,int,int,bool,int,bool]] = [
            (3,  16,  16, False, 1, False),
            (3,  48,  24, False, 2, False),
            (3,  72,  24, False, 1, False),
            (5,  72,  40, True,  2, False),
            (5, 120,  40, True,  1, False),
            (3, 240,  80, False, 2, True ),
            (3, 200,  80, False, 1, True ),
            (3, 184,  80, False, 1, True ),
            (3, 184,  80, False, 1, True ),
            (3, 480, 112, True,  1, True ),
            (3, 672, 112, True,  1, True ),
            (5, 672, 160, True,  2, True ),
            (5, 960, 160, True,  1, True ),
            (5, 960, 160, True,  1, True ),
        ]

        layers: List[nn.Module] = []
        for k, exp, out, se, s, hs in cfg:
            layers.append(GhostBottleneck(in_ch, c(exp), c(out), kernel=k, stride=s, use_se=se, act_hs=hs, ratio=ghost_ratio))
            in_ch = c(out)
        self.blocks = nn.Sequential(*layers)

        # Head
        head_ch = c(960)
        self.head = nn.Sequential(
            nn.Conv2d(in_ch, head_ch, 1, bias=False),
            nn.BatchNorm2d(head_ch),
            HSwish(inplace=True),
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


def make_ghostnet_cifar(num_classes: int = 10, in_channels: int = 3, width_mult: float = 1.0, ghost_ratio: int = 2, dropout: float = 0.0) -> GhostNetCIFAR:
    return GhostNetCIFAR(num_classes=num_classes, in_channels=in_channels, width_mult=width_mult, ghost_ratio=ghost_ratio, dropout=dropout)

if __name__ == "__main__":
    m = make_ghostnet_cifar(num_classes=10, width_mult=1.0, ghost_ratio=2)
    y = m(torch.randn(2,3,32,32))
    print(y.shape, GhostNetCIFAR.param_count(m))
