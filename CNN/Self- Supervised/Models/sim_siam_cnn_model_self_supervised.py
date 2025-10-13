"""
SimSiam with a lightweight CNN encoder (self-supervised, PyTorch)
------------------------------------------------------------------
Components
  • ConvEncoder: configurable CNN backbone (Conv-BN-ReLU blocks, optional MaxPool, GAP)
  • ProjectionMLP: 2-layer MLP head
  • PredictionMLP: 2-layer MLP predictor (bottleneck)
  • SimSiam: symmetric stop-gradient loss (no momentum/EMA networks)

Conventions
  • Pool index convention is 0-based (e.g., [0, 2] means pool after blocks 0 and 2)
  • Outputs for loss are L2-normalized
  • total_neurons() returns parameter-count proxy

Loss
  • SimSiam uses symmetric negative cosine between predictions and stop-grad projections:
      L = (2 - 2·cos(p1, stopgrad(z2)) + 2 - 2·cos(p2, stopgrad(z1))) / 2
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
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x


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
            x = x.mean(dim=(-2, -1))  # global average pooling
        return x

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        f = self.forward(x)
        f = F.normalize(f, dim=-1)
        return f


class ProjectionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, out_dim: int = 2048):
        super().__init__()
        hidden_dim = hidden_dim or (2 * in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z


class PredictionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, out_dim: Optional[int] = None):
        super().__init__()
        # SimSiam predictor is usually smaller; default hidden = in_dim // 2
        hidden_dim = hidden_dim or max(64, in_dim // 2)
        out_dim = out_dim or in_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.net(x)
        p = F.normalize(p, dim=-1)
        return p


def _neg_cosine(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    z = z.detach()  # stop-grad on target
    return 2.0 - 2.0 * (p * z).sum(dim=-1)


class SimSiam(nn.Module):
    def __init__(
        self,
        encoder: ConvEncoder,
        proj_hidden: Optional[int] = None,
        proj_dim: int = 2048,
        pred_hidden: Optional[int] = None,
    ):
        super().__init__()
        self.encoder = encoder
        self.proj = ProjectionMLP(in_dim=encoder.out_dim, hidden_dim=proj_hidden, out_dim=proj_dim)
        self.pred = PredictionMLP(in_dim=proj_dim, hidden_dim=pred_hidden, out_dim=proj_dim)

    def forward_once(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f = self.encoder(x)   # [B, D]
        z = self.proj(f)      # [B, P] (normalized)
        p = self.pred(z)      # [B, P] (normalized)
        return p, z

    def loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        p1, z1 = self.forward_once(x1)
        p2, z2 = self.forward_once(x2)
        loss = _neg_cosine(p1, z2) + _neg_cosine(p2, z1)
        return loss.mean()

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
    "ProjectionMLP",
    "PredictionMLP",
    "SimSiam",
    "total_neurons",
]
