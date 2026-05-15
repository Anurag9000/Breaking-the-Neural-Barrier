"""
ShuffleNet V1 (CIFAR) — single-model supervised.

Implements ShuffleNet (Zhang et al., 2018) with **group 1×1 convs**, **channel shuffle**, and
**depthwise 3×3**. CIFAR-friendly: stem uses stride=1; stage downsampling on first block of
stages 2 and 3 (→ feature map sizes 32→16→8→4). GAP→FC head.

Config mirrors common ImageNet setup adapted to CIFAR:
  • Stage repeats: [4, 8, 4]
  • Stage output channels depend on groups (paper presets), with optional width multiplier.

Factory exposes groups (g∈{1,2,3,4,8}) and width_mult, plus param_count helper.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

__all__ = [
    "ShuffleNetV1CIFAR",
    "make_shufflenet_v1_cifar",
]

# Preset stage out channels per paper (including outputs of stages 1..3; stem uses 24)
_STAGE_OUT = {
    1: [144, 288, 576],
    2: [200, 400, 800],
    3: [240, 480, 960],
    4: [272, 544, 1088],
    8: [384, 768, 1536],
}

def _make_divisible(v: int | float, divisor: int = 8) -> int:
    return max(divisor, int(v + divisor / 2) // divisor * divisor)

def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    N, C, H, W = x.size()
    assert C % groups == 0
    g = groups
    x = x.view(N, g, C // g, H, W)
    x = x.transpose(1, 2).contiguous()
    x = x.view(N, C, H, W)
    return x

class ConvBNAct(nn.Sequential):
    def __init__(self, in_ch: int, out_ch: int, k: int, s: int = 1, p: int | None = None, groups: int = 1, act: str = "relu"):
        if p is None:
            p = (k - 1) // 2
        act_layer = nn.ReLU(inplace=True) if act == "relu" else nn.Identity()
        super().__init__(
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, groups=groups, bias=False),
            nn.BatchNorm2d(out_ch),
            act_layer,
        )

class ShuffleUnit(nn.Module):
    """ShuffleNet unit (Res-Unit style).
    - 1×1 GConv (BN, ReLU)
    - Channel Shuffle
    - 3×3 DWConv (BN)
    - 1×1 GConv (BN) (no activation at end)
    - Residual: identity if stride=1 else AvgPool and channel concat (as in paper)
    """
    def __init__(self, in_ch: int, out_ch: int, stride: int, groups: int):
        super().__init__()
        assert stride in (1, 2)
        self.stride = stride
        mid_ch = out_ch // 4
        # First 1×1 group conv (paper uses g=1 for the first stage when in_ch == 24)
        g1 = 1 if in_ch == 24 else groups
        self.gconv1 = ConvBNAct(in_ch, mid_ch, k=1, s=1, p=0, groups=g1, act="relu")
        # DW 3×3
        self.dwconv = ConvBNAct(mid_ch, mid_ch, k=3, s=stride, groups=mid_ch, act="none")
        # Second 1×1 group conv (linear)
        out_gconv2 = out_ch if stride == 1 else out_ch - in_ch
        self.gconv2 = ConvBNAct(mid_ch, out_gconv2, k=1, s=1, p=0, groups=groups, act="none")
        self.relu = nn.ReLU(inplace=True)
        if stride == 2:
            self.shortcut = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        else:
            self.shortcut = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.gconv1(x)
        # groups of second gconv define shuffle groups
        g = self.gconv2[0].groups if hasattr(self.gconv2[0], 'groups') else 1
        out = channel_shuffle(out, groups=g)
        out = self.dwconv(out)
        out = self.gconv2(out)
        if self.stride == 1:
            out = x + out
        else:
            out = torch.cat([self.shortcut(x), out], dim=1)
        out = self.relu(out)
        return out

class ShuffleNetV1CIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, groups: int = 3, width_mult: float = 1.0):
        super().__init__()
        if groups not in _STAGE_OUT:
            raise ValueError(f"groups must be one of {list(_STAGE_OUT.keys())}")
        base_out = _STAGE_OUT[groups]
        out_channels = [24] + [ _make_divisible(c * width_mult, 8) for c in base_out ]
        # Stem
        self.conv1 = ConvBNAct(in_channels, 24, k=3, s=1)
        # Stages: repeats per paper
        repeats = [4, 8, 4]
        self.stage2 = self._make_stage(24, out_channels[1], repeats[0], groups, downsample=True)
        self.stage3 = self._make_stage(out_channels[1], out_channels[2], repeats[1], groups, downsample=True)
        self.stage4 = self._make_stage(out_channels[2], out_channels[3], repeats[2], groups, downsample=False)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(out_channels[3], num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def _make_stage(self, in_ch: int, out_ch: int, repeats: int, groups: int, downsample: bool) -> nn.Sequential:
        layers: List[nn.Module] = []
        # first block may downsample (stride=2) when downsample=True
        layers.append(ShuffleUnit(in_ch, out_ch, stride=2 if downsample else 1, groups=groups))
        for _ in range(repeats - 1):
            layers.append(ShuffleUnit(out_ch, out_ch, stride=1, groups=groups))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_shufflenet_v1_cifar(num_classes: int = 10, in_channels: int = 3, groups: int = 3, width_mult: float = 1.0) -> ShuffleNetV1CIFAR:
    return ShuffleNetV1CIFAR(num_classes=num_classes, in_channels=in_channels, groups=groups, width_mult=width_mult)
