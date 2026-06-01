"""
Barlow Twins with a lightweight CNN encoder (self-supervised, PyTorch)
---------------------------------------------------------------------
Components
  • ConvEncoder: configurable CNN backbone (Conv-BN-ReLU blocks, optional MaxPool, GAP)
  • ProjectorMLP: 3-layer MLP (as per Barlow Twins) projecting to high-dim space
  • barlow_twins_loss: cross-correlation decorrelation objective

Conventions
  • Pool index convention is 0-based (e.g., [0, 2] means pool after blocks 0 and 2)
  • Embeddings for the loss are batch-normalized (zero mean, unit variance)
  • total_neurons() returns parameter-count proxy

Loss (Barlow Twins)
  Let z1, z2 be projector outputs from two augmented views (shape: [B, D]).
  Normalize each dimension with batch statistics, then compute the cross-correlation matrix C = (z1ᵀ z2) / B.
  L = Σ_i (1 - C_ii)^2  +  λ Σ_{i≠j} C_ij^2
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


class ProjectorMLP(nn.Module):
    """3-layer MLP projector with BN (last BN without affine as in Barlow Twins)."""
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, out_dim: int = 2048):
        super().__init__()
        h = hidden_dim or (2 * in_dim)
        self.layer1 = nn.Sequential(
            nn.Linear(in_dim, h, bias=True),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
        )
        self.layer2 = nn.Sequential(
            nn.Linear(h, h, bias=True),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Linear(h, out_dim, bias=False),
            nn.BatchNorm1d(out_dim, affine=False),  # BN w/o affine, as per paper
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x  # not L2-normalized; BN already applied


class BarlowTwins(nn.Module):
    def __init__(self, encoder: ConvEncoder, proj_hidden: Optional[int] = None, proj_dim: int = 2048):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectorMLP(in_dim=encoder.out_dim, hidden_dim=proj_hidden, out_dim=proj_dim)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        return z1, z2


# ----------------------------
# Loss: Barlow Twins
# ----------------------------

def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    off = x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    return off


def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor, lambd: float = 5e-3) -> torch.Tensor:
    """Compute Barlow Twins loss.

    Args:
        z1, z2: [B, D] projector outputs from two views
        lambd: off-diagonal scaling (lambda)
    Returns:
        Scalar loss
    """
    assert z1.dim() == 2 and z2.dim() == 2 and z1.shape == z2.shape
    B, D = z1.shape

    # Normalize each dimension with batch statistics
    z1 = (z1 - z1.mean(dim=0)) / (z1.std(dim=0) + 1e-9)
    z2 = (z2 - z2.mean(dim=0)) / (z2.std(dim=0) + 1e-9)

    # Cross-correlation matrix: [D, D]
    c = (z1.T @ z2) / B

    on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
    off_diag = _off_diagonal(c).pow_(2).sum()
    loss = on_diag + lambd * off_diag
    return loss


# ----------------------------
# Utility: parameter-count proxy
# ----------------------------

def total_neurons(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "ConvEncoder",
    "ProjectorMLP",
    "BarlowTwins",
    "barlow_twins_loss",
    "total_neurons",
]
