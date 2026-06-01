"""
RepLKNet (CIFAR) — single-model supervised.

Lightweight CIFAR adaptation of RepLKNet (Ding et al., 2022):
  • RepLK Block: 1×1 conv → Large‑kernel **depthwise** conv (k≳13 on CIFAR) → 1×1 conv,
    with LayerNorm2d, GELU, residual, optional LayerScale γ and DropPath.
  • Stage downsampling by 2×2 stride‑2 conv (32→16→8→4).
  • Head: GAP → Linear.

Presets expose depths/dims and default large kernel size; configurable via args.
This follows your factory/run patterns and stays strictly single‑model supervised.
"""
from __future__ import annotations
from typing import List
import torch
import torch.nn as nn

__all__ = [
    "RepLKNetCIFAR",
    "make_replknet_cifar",
]

# ----------------------
# Utils
# ----------------------
class LayerNorm2d(nn.Module):
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)
        return x

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

# ----------------------
# RepLK Block
# ----------------------
class RepLKBlock(nn.Module):
    def __init__(self, dim: int, k: int = 13, layer_scale_init: float = 1e-6, drop_path: float = 0.0):
        super().__init__()
        pad = k // 2
        self.norm1 = LayerNorm2d(dim)
        self.pw1 = nn.Conv2d(dim, dim, kernel_size=1)
        self.dw = nn.Conv2d(dim, dim, kernel_size=k, padding=pad, groups=dim)
        self.act = nn.GELU()
        self.pw2 = nn.Conv2d(dim, dim, kernel_size=1)
        self.norm2 = LayerNorm2d(dim)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones(dim)) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.norm1(x)
        x = self.pw1(x)
        x = self.dw(x)
        x = self.act(x)
        x = self.pw2(x)
        x = self.norm2(x)
        if self.gamma is not None:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = self.drop_path(x)
        return shortcut + x

# ----------------------
# Network
# ----------------------
PRESETS = {
    # depths, dims, large kernel
    "tiny":  dict(depths=[2, 2, 6, 2], dims=[64,  128, 256, 512], k=13),
    "small": dict(depths=[3, 3, 9,  3], dims=[80,  160, 320, 640], k=15),
    "base":  dict(depths=[3, 4, 12, 3], dims=[96,  192, 384, 768], k=17),
}

class RepLKNetCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3, preset: str = "tiny",
                 layer_scale_init: float = 1e-6, drop_path_rate: float = 0.0, k_override: int | None = None):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS.keys())}")
        cfg = PRESETS[preset]
        depths = cfg["depths"]
        dims = cfg["dims"]
        k_lk = k_override if k_override is not None else cfg["k"]

        # Stem (stride=1)
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, dims[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(dims[0]),
            nn.ReLU(inplace=True),
        )

        # Downsample layers between stages
        self.downsample = nn.ModuleList([
            nn.Identity(),
            nn.Sequential(LayerNorm2d(dims[0]), nn.Conv2d(dims[0], dims[1], kernel_size=2, stride=2)),
            nn.Sequential(LayerNorm2d(dims[1]), nn.Conv2d(dims[1], dims[2], kernel_size=2, stride=2)),
            nn.Sequential(LayerNorm2d(dims[2]), nn.Conv2d(dims[2], dims[3], kernel_size=2, stride=2)),
        ])

        # Stages
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks: List[nn.Module] = []
            for j in range(depths[i]):
                blocks.append(RepLKBlock(dims[i], k=k_lk, layer_scale_init=layer_scale_init, drop_path=dp_rates[cur + j]))
            cur += depths[i]
            self.stages.append(nn.Sequential(*blocks))

        # Head
        self.head_norm = LayerNorm2d(dims[-1])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(dims[-1], num_classes)
        nn.init.trunc_normal_(self.fc.weight, std=0.02)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stages[0](x)
        for i in range(1, 4):
            x = self.downsample[i](x)
            x = self.stages[i](x)
        x = self.head_norm(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_replknet_cifar(num_classes: int = 10, in_channels: int = 3, preset: str = "tiny",
                         layer_scale_init: float = 1e-6, drop_path_rate: float = 0.0, k_override: int | None = None) -> RepLKNetCIFAR:
    return RepLKNetCIFAR(num_classes=num_classes, in_channels=in_channels, preset=preset,
                         layer_scale_init=layer_scale_init, drop_path_rate=drop_path_rate, k_override=k_override)
