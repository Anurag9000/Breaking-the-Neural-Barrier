"""
ConvNeXt (CIFAR) — single-model supervised.

Implements ConvNeXt (Liu et al., 2022) with CIFAR-friendly tweaks:
  • Stem 4→ (for CIFAR use stride=1 first downsample to keep 32→32, then stages downsample: 32→16→8→4)
  • ConvNeXt block: DWConv 7×7 → LayerNorm (channels-last) → MLP (Linear 4×, GELU, Linear) → LayerScale → residual
  • Stochastic Depth (DropPath) per block

Exposes model size via preset {tiny, small, base} (depths/dims) and width_mult.
Head: global average pooling → linear classifier.

Notes:
  • LayerNorm is applied in NHWC for numerical parity with the paper; we permute tensors.
  • LayerScale (gamma) default 1e-6 for tiny/small/base, configurable.
"""
from __future__ import annotations
from typing import List, Tuple
import torch
import torch.nn as nn

__all__ = [
    "ConvNeXtCIFAR",
    "make_convnext_cifar",
]

# ----------------------
# Utils
# ----------------------
class LayerNorm2d(nn.Module):
    """LayerNorm over channel dimension for NCHW by temporarily permuting to NHWC."""
    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.ln = nn.LayerNorm(num_channels, eps=eps)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 3, 1)  # NCHW -> NHWC
        x = self.ln(x)
        x = x.permute(0, 3, 1, 2)  # NHWC -> NCHW
        return x

class DropPath(nn.Module):
    """Stochastic Depth per sample (when applied in main path)."""
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        # Work with shape (N, 1, 1, 1)
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        return x / keep_prob * random_tensor

# ----------------------
# ConvNeXt Block
# ----------------------
class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, layer_scale_init: float = 1e-6, drop_path: float = 0.0):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm2d(dim, eps=1e-6)
        self.pwconv1 = nn.Conv2d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv2d(4 * dim, dim, kernel_size=1)
        self.gamma = nn.Parameter(layer_scale_init * torch.ones((dim)), requires_grad=True) if layer_scale_init > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.dwconv(x)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            # apply per-channel gamma (NCHW)
            x = x * self.gamma.view(1, -1, 1, 1)
        x = self.drop_path(x)
        x = x + shortcut
        return x

# ----------------------
# Network
# ----------------------
PRESETS = {
    # depths, dims
    "tiny":  dict(depths=[3, 3, 9, 3], dims=[96, 192, 384, 768]),
    "small": dict(depths=[3, 3, 27, 3], dims=[96, 192, 384, 768]),
    "base":  dict(depths=[3, 3, 27, 3], dims=[128, 256, 512, 1024]),
}

class ConvNeXtCIFAR(nn.Module):
    def __init__(self, num_classes: int = 10, in_channels: int = 3,
                 preset: str = "tiny", width_mult: float = 1.0,
                 layer_scale_init: float = 1e-6, drop_path_rate: float = 0.0):
        super().__init__()
        if preset not in PRESETS:
            raise ValueError(f"preset must be one of {list(PRESETS.keys())}")
        cfg = PRESETS[preset]
        depths: List[int] = cfg["depths"]
        dims: List[int] = [max(8, int(d * width_mult)) for d in cfg["dims"]]

        # CIFAR stem: keep stride=1 at first stage; then downsample at stage transitions
        self.downsample_layers = nn.ModuleList()
        # stem
        stem_dim = dims[0]
        stem = nn.Sequential(
            nn.Conv2d(in_channels, stem_dim, kernel_size=3, stride=1, padding=1),
            LayerNorm2d(stem_dim)
        )
        self.downsample_layers.append(stem)
        # stage downsample layers (3 transitions)
        for i in range(3):
            in_dim = dims[i]
            out_dim = dims[i + 1]
            down = nn.Sequential(
                LayerNorm2d(in_dim),
                nn.Conv2d(in_dim, out_dim, kernel_size=2, stride=2)
            )
            self.downsample_layers.append(down)

        # stages
        dp_rates = torch.linspace(0, drop_path_rate, sum(depths)).tolist()
        cur = 0
        self.stages = nn.ModuleList()
        for i in range(4):
            blocks = []
            for j in range(depths[i]):
                blocks.append(ConvNeXtBlock(dims[i], layer_scale_init=layer_scale_init, drop_path=dp_rates[cur + j]))
            cur += depths[i]
            self.stages.append(nn.Sequential(*blocks))

        self.head_norm = LayerNorm2d(dims[-1])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(dims[-1], num_classes)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.downsample_layers[0](x)
        x = self.stages[0](x)
        for i in range(1, 4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        x = self.head_norm(x)
        x = self.gap(x)
        x = torch.flatten(x, 1)
        x = self.head(x)
        return x

    @staticmethod
    def param_count(model: nn.Module) -> int:
        return sum(p.numel() for p in model.parameters())


def make_convnext_cifar(num_classes: int = 10, in_channels: int = 3, preset: str = "tiny",
                         width_mult: float = 1.0, layer_scale_init: float = 1e-6, drop_path_rate: float = 0.0) -> ConvNeXtCIFAR:
    return ConvNeXtCIFAR(num_classes=num_classes, in_channels=in_channels, preset=preset,
                         width_mult=width_mult, layer_scale_init=layer_scale_init, drop_path_rate=drop_path_rate)

if __name__ == "__main__":
    m = make_convnext_cifar(num_classes=10, preset="tiny", drop_path_rate=0.1)
    y = m(torch.randn(2,3,32,32))
    print(y.shape, ConvNeXtCIFAR.param_count(m))
