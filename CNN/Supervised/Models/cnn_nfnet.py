"""
NFNet (CIFAR) — single-model supervised (Normalizer-Free Network).

A compact, CIFAR-friendly NFNet with **Scaled Weight Standardization (SWS)**, **no normalization layers**,
SiLU/GELU activations, optional **SE**, residual scaling (β) and DropPath.

Design (per block):
  x ──► ConvWS(1×1) ─ Act ─ ConvWS(3×3, stride s) ─ Act ─ ConvWS(1×1) ─ SE? ──┐
   └─────────────────────────────── downsample (if needed) ──────────────┘   │
                                (residual add with β, then Act)             ▼

CIFAR tweaks:
  • Stem is 3×3 s=1.
  • Stage downsample at the first block of stages 2–4 → 32→16→8→4.

Presets (lite): {L0, L1, L2} with depths & widths inspired by NFNet-F/L families but reduced.
Factory mirrors your style and includes param_count helper.
"""
from __future__ import annotations
from typing import List
import math
import torch
import torch.nn as nn

__all__ = [
    "NFNetCIFAR",
    "make_nfnet_cifar",
]

# ----------------------
# Utils
# ----------------------
class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        return x * x.new_empty(shape).bernoulli_(keep) / keep

class SEModule(nn.Module):
    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(channels // reduction, 8)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, hidden, 1)
        self.act = nn.SiLU(inplace=True)
        self.fc2 = nn.Conv2d(hidden, channels, 1)
        self.gate = nn.Sigmoid()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.pool(x)
        w = self.act(self.fc1(w))
        w = self.gate(self.fc2(w))
        return x * w

# ----------------------
# Scaled Weight Standardized Conv
# ----------------------
class Conv2dWS(nn.Conv2d):
    """Conv2d with per-output-channel Weight Standardization and a learnable scale (gain).
    Uses the NFNet SWS scaling factor: ŵ = (w - mean) / std; out = conv(x, s * (g ⊙ ŵ)),
    where s = gamma / sqrt(fan_in), and g is a per-output-channel gain parameter (initialized to 1).
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, gamma: float = 1.0, eps: float = 1e-5):
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.eps = eps
        self.gain = nn.Parameter(torch.ones(out_channels))
        # scale constant s following SWS: gamma/sqrt(fan_in * k*k/groups)
        k = self.weight.shape[2]
        fan_in = (in_channels // groups) * (k * k)
        self.register_buffer("ws_scale", torch.tensor(gamma / math.sqrt(fan_in), dtype=self.weight.dtype))

    def _ws_weight(self) -> torch.Tensor:
        w = self.weight
        # per-out-channel mean/std over (in, k, k)
        mean = w.mean(dim=(1,2,3), keepdim=True)
        w_centered = w - mean
        std = w_centered.flatten(1).std(dim=1, keepdim=True).view(-1,1,1,1)
        w_norm = w_centered / (std + self.eps)
        scale = self.ws_scale * self.gain.view(-1, 1, 1, 1)
        return w_norm * scale

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nn.functional.conv2d(x, self._ws_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

# ----------------------
# NFNet Block
# ----------------------
class NFBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, beta: float = 1.0, se_ratio: float = 0.0,
                 drop_path: float = 0.0, gamma: float = 1.0, act: str = "silu"):
        super().__init__()
        Act = nn.SiLU if act == "silu" else nn.GELU
        mid = out_ch
        self.conv1 = Conv2dWS(in_ch, mid, kernel_size=1, stride=1, padding=0, bias=True, gamma=gamma)
        self.act1 = Act(inplace=True)
        self.conv2 = Conv2dWS(mid, mid, kernel_size=3, stride=stride, padding=1, groups=1, bias=True, gamma=gamma)
        self.act2 = Act(inplace=True)
        self.conv3 = Conv2dWS(mid, out_ch, kernel_size=1, stride=1, padding=0, bias=True, gamma=gamma)
        self.se = SEModule(out_ch, reduction=int(1/se_ratio)) if se_ratio and se_ratio > 0 else nn.Identity()
        # skip / downsample
        if stride != 1 or in_ch != out_ch:
            self.skip = Conv2dWS(in_ch, out_ch, kernel_size=1, stride=stride, padding=0, bias=True, gamma=gamma)
        else:
            self.skip = nn.Identity()
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.beta = beta
        self.act_out = Act(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.conv1(x); y = self.act1(y)
        y = self.conv2(y); y = self.act2(y)
        y = self.conv3(y)
        y = self.se(y)
        y = self.drop_path(y)
        out = self.skip(x) + self.beta * y
        return self.act_out(out)

# ----------------------
# Network
# ----------------------
PRESETS = {
    # depths per stage [s1,s2,s3,s4], widths per stage, se_ratio
    "L0": dict(depths=[2, 4, 8, 2], widths=[64, 128, 256, 512], se=0.5),
    "L1": dict(depths=[2, 6, 12, 2], widths=[96, 192, 384, 768], se=0.5),
    "L2": dict(depths=[3, 6, 12, 3], widths=[128, 256, 512, 1024], se=0.5),
}

class NFNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, preset: str = "L0",
                 beta: float = 1.0, drop_path_rate: float = 0.0, gamma: float = 1.0, act: str = "silu"):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS.keys())}")
        cfg = PRESETS[preset]
        depths = cfg["depths"]
        widths = cfg["widths"]
        se_ratio = cfg["se"]

        # Stem
        self.stem = nn.Sequential(
            Conv2dWS(in_channels, widths[0], kernel_size=3, stride=1, padding=1, bias=True, gamma=gamma),
            nn.SiLU(inplace=True) if act=="silu" else nn.GELU(),
        )

        # Stages (downsample at start of stages 2–4)
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        in_ch = widths[0]
        self.stages = nn.ModuleList()
        for si in range(4):
            blocks: List[nn.Module] = []
            stride_first = 1 if si == 0 else 2
            out_ch = widths[si]
            # first block (maybe downsample)
            blocks.append(NFBlock(in_ch, out_ch, stride=stride_first, beta=beta, se_ratio=se_ratio,
                                  drop_path=dp_rates[cur], gamma=gamma, act=act))
            cur += 1
            in_ch = out_ch
            # remaining blocks
            for _ in range(depths[si] - 1):
                blocks.append(NFBlock(in_ch, out_ch, stride=1, beta=beta, se_ratio=se_ratio,
                                      drop_path=dp_rates[cur], gamma=gamma, act=act))
                cur += 1
            self.stages.append(nn.Sequential(*blocks))

        # Head
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(in_ch, num_classes)
        nn.init.kaiming_normal_(self.fc.weight, nonlinearity='relu')
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        for s in self.stages:
            x = s(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_nfnet_cifar(num_classes: int = 10, in_channels: int = 3, preset: str = "L0",
                      beta: float = 1.0, drop_path_rate: float = 0.0, gamma: float = 1.0, act: str = "silu") -> NFNetCIFAR:
    return NFNetCIFAR(num_classes=num_classes, in_channels=in_channels, preset=preset,
                      beta=beta, drop_path_rate=drop_path_rate, gamma=gamma, act=act)

if __name__ == "__main__":
    m = make_nfnet_cifar(num_classes=10, preset="L0", drop_path_rate=0.1)
    y = m(torch.randn(2,3,32,32))
    print(y.shape, NFNetCIFAR.param_count(m))
