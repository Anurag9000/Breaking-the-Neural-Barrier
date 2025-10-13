"""
SwAV with a lightweight CNN encoder (self-supervised, PyTorch)
---------------------------------------------------------------
Core ideas
  • Instance features are assigned to prototype codes via online Sinkhorn-Knopp
  • Learn by matching codes across augmented views (swapped assignments)

Components
  • ConvEncoder: configurable CNN backbone (Conv-BN-ReLU blocks, optional MaxPool, GAP)
  • ProjectionMLP: 2-layer MLP head before prototypes
  • Prototypes: learnable codebook (weight-normalized linear layer)
  • sinkhorn_knopp: balanced assignments within a batch (no memory queue in this minimal version)

Conventions
  • Pool index convention: 0-based (e.g., [0, 2] => pool after blocks 0 and 2)
  • Features and projection outputs are L2-normalized before prototypes
  • total_neurons() returns parameter-count proxy

Notes
  • This is a compact two-view SwAV (no multi-crop); extend with extra views if desired.
  • Temperature params: τ_p for prototypes (softmax logits), τ_s for swapped cross-entropy.
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
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, out_dim: int = 128):
        super().__init__()
        hidden_dim = hidden_dim or (2 * in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z


class Prototypes(nn.Module):
    """Weight-normalized prototypes (codebook)."""
    def __init__(self, in_dim: int, n_prototypes: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(n_prototypes, in_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, z: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
        # logits: [B, K]
        w = F.normalize(self.weight, dim=1)  # normalize codes
        z = F.normalize(z, dim=1)
        return (z @ w.t()) / max(1e-8, temperature)


# ----------------------------
# Sinkhorn-Knopp for balanced assignments
# ----------------------------
@torch.no_grad()
def sinkhorn_knopp(logits: torch.Tensor, epsilon: float = 0.05, n_iters: int = 3) -> torch.Tensor:
    """Return soft assignments Q that sum to 1 across prototypes and samples.
       logits: [B, K]
    """
    Q = torch.exp(logits / epsilon).t()  # [K, B]
    B = Q.shape[1]
    K = Q.shape[0]
    Q /= Q.sum()
    for _ in range(n_iters):
        # normalize rows
        Q /= Q.sum(dim=1, keepdim=True)
        Q /= K
        # normalize cols
        Q /= Q.sum(dim=0, keepdim=True)
        Q /= B
    Q *= B  # columns sum to 1
    return Q.t().contiguous()  # [B, K]


class SwAV(nn.Module):
    def __init__(
        self,
        encoder: ConvEncoder,
        proj_hidden: Optional[int] = None,
        proj_dim: int = 128,
        n_prototypes: int = 300,
        tau_p: float = 0.1,
        tau_s: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionMLP(in_dim=encoder.out_dim, hidden_dim=proj_hidden, out_dim=proj_dim)
        self.prototypes = Prototypes(in_dim=proj_dim, n_prototypes=n_prototypes)
        self.tau_p = tau_p
        self.tau_s = tau_s

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Project
        f1 = self.encoder(x1)
        f2 = self.encoder(x2)
        z1 = self.projector(f1)
        z2 = self.projector(f2)
        # Prototype logits (unnormalized assignments)
        p1 = self.prototypes(z1, temperature=self.tau_p)  # [B, K]
        p2 = self.prototypes(z2, temperature=self.tau_p)  # [B, K]
        return z1, z2, p1, p2

    def loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        z1, z2, p1, p2 = self.forward(x1, x2)
        # Balanced soft assignments via Sinkhorn
        with torch.no_grad():
            q1 = sinkhorn_knopp(p1.detach())  # [B, K]
            q2 = sinkhorn_knopp(p2.detach())
        # Swapped prediction cross-entropy
        l1 = -(q1 * F.log_softmax(p2 / max(1e-8, self.tau_s), dim=1)).sum(dim=1).mean()
        l2 = -(q2 * F.log_softmax(p1 / max(1e-8, self.tau_s), dim=1)).sum(dim=1).mean()
        return (l1 + l2) * 0.5

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)


# ----------------------------
# Utility: parameter-count proxy
# ----------------------------
import math

def total_neurons(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "ConvEncoder",
    "ProjectionMLP",
    "Prototypes",
    "SwAV",
    "sinkhorn_knopp",
    "total_neurons",
]
