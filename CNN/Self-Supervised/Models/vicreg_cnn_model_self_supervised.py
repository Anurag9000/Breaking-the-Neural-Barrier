"""
VICReg with a lightweight CNN encoder (self-supervised, PyTorch)
-----------------------------------------------------------------
Components
  • ConvEncoder: configurable CNN backbone (Conv-BN-ReLU blocks, optional MaxPool, GAP)
  • ProjectorMLP: 3-layer MLP (as in VICReg/BT) projecting to high-dim space
  • vicreg_loss: variance–invariance–covariance regularization objective

Conventions
  • Pool index convention: 0-based (e.g., [0, 2] => pool after blocks 0 and 2)
  • total_neurons() returns parameter-count proxy

Loss (VICReg)
  Let z1, z2 be projector outputs for two augmented views: shape [B, D].
  1) Invariance: L_inv = MSE(z1, z2)
  2) Variance:  enforce per-dim std >= γ via hinge on (γ - std)+ for each view
  3) Covariance: penalize off-diagonal covariance within each view
  L = α·L_inv + β·(Var(z1)+Var(z2)) + μ·(CovOff(z1)+CovOff(z2))
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
        f = self.forward(x)
        return F.normalize(f, dim=-1)


class ProjectorMLP(nn.Module):
    """3-layer MLP projector with BN (last BN without affine) as in VICReg/BT."""
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, out_dim: int = 2048):
        super().__init__()
        h = hidden_dim or (2 * in_dim)
        self.layer1 = nn.Sequential(nn.Linear(in_dim, h, bias=True), nn.BatchNorm1d(h), nn.ReLU(inplace=True))
        self.layer2 = nn.Sequential(nn.Linear(h, h, bias=True), nn.BatchNorm1d(h), nn.ReLU(inplace=True))
        self.layer3 = nn.Sequential(nn.Linear(h, out_dim, bias=False), nn.BatchNorm1d(out_dim, affine=False))
        self.out_dim = out_dim
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layer3(self.layer2(self.layer1(x)))


class VICReg(nn.Module):
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
# VICReg loss
# ----------------------------

def _off_diagonal(x: torch.Tensor) -> torch.Tensor:
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


def vicreg_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    alpha: float = 25.0,
    beta: float = 25.0,
    mu: float = 1.0,
    gamma: float = 1.0,
) -> torch.Tensor:
    """Variance–Invariance–Covariance Regularization.
    Args:
        z1, z2: [B, D] projector outputs from two views
        alpha: invariance (MSE) weight
        beta: variance weight
        mu: covariance off-diagonal weight
        gamma: target std per dimension
    """
    assert z1.dim() == 2 and z2.dim() == 2 and z1.shape == z2.shape
    B, D = z1.shape
    # Invariance (MSE)
    l_inv = F.mse_loss(z1, z2)
    # Variance (per-dim std hinge), compute for each view separately
    def _var(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0)
        std = torch.sqrt(z.var(dim=0) + 1e-4)
        return torch.mean(F.relu(gamma - std))
    l_var = _var(z1) + _var(z2)
    # Covariance (off-diagonal) per view
    def _cov_off(z: torch.Tensor) -> torch.Tensor:
        z = z - z.mean(dim=0)
        cov = (z.T @ z) / (B - 1)  # [D, D]
        off = _off_diagonal(cov)
        return (off.pow(2).sum()) / D
    l_cov = _cov_off(z1) + _cov_off(z2)
    return alpha * l_inv + beta * l_var + mu * l_cov


# ----------------------------
# Utility: parameter-count proxy
# ----------------------------

def total_neurons(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "ConvEncoder",
    "ProjectorMLP",
    "VICReg",
    "vicreg_loss",
    "total_neurons",
]
