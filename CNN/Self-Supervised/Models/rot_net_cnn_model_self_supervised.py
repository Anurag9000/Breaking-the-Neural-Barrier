"""
RotNet with a lightweight CNN encoder (self-supervised, PyTorch)
-----------------------------------------------------------------
Task
  • Predict which rotation (0°, 90°, 180°, 270°) was applied to an input image.

Components
  • ConvEncoder: configurable CNN backbone (Conv-BN-ReLU blocks, optional MaxPool, GAP)
  • RotationHead: linear classifier over 4 rotation classes

Usage pattern
  • Sample an image, create 4 rotated copies, train with cross-entropy to predict angle.
  • After pretraining, discard the head and keep the encoder for downstream tasks.

Conventions
  • Pool index convention: 0-based (e.g., [0, 2] => pool after blocks 0 and 2)
  • total_neurons() returns parameter-count proxy
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Building blocks
# ----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class ConvEncoder(nn.Module):
    """Configurable CNN encoder with optional MaxPool after specified 0-based blocks and GAP head."""
    def __init__(
        self,
        in_ch: int = 3,
        width: int = 64,
        depth: int = 4,
        pool_after: Optional[List[int]] = None,
        gap: bool = True,
    ):
        super().__init__()
        assert depth >= 1
        pool_after = pool_after or []
        layers: List[nn.Module] = []
        ch_in = in_ch
        for i in range(depth):
            layers.append(ConvBNReLU(ch_in, width, k=3, s=1))
            ch_in = width
            if i in set(pool_after):
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        self.features = nn.Sequential(*layers)
        self.gap = gap
        self.out_dim = width
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.gap:
            x = x.mean(dim=(-2, -1))
        return x
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.forward(x), dim=-1)


class RotationHead(nn.Module):
    def __init__(self, in_dim: int, num_classes: int = 4):
        super().__init__()
        self.fc = nn.Linear(in_dim, num_classes)
        self.num_classes = num_classes
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class RotNet(nn.Module):
    def __init__(self, encoder: ConvEncoder, num_classes: int = 4):
        super().__init__()
        self.encoder = encoder
        self.head = RotationHead(in_dim=encoder.out_dim, num_classes=num_classes)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        logits = self.head(f)
        return logits
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)


# ----------------------------
# Utility: parameter-count proxy
# ----------------------------

def total_neurons(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "ConvEncoder",
    "RotationHead",
    "RotNet",
    "total_neurons",
]
