"""
SimCLR with a lightweight CNN encoder (self-supervised, PyTorch)
-----------------------------------------------------------------
This module provides:
  • ConvEncoder: configurable CNN backbone (width, depth, optional maxpools after given blocks)
  • ProjectionHead: 2-layer MLP for SimCLR projection
  • SimCLR: wrapper that holds encoder + projection head
  • nt_xent_loss: the normalized temperature-scaled cross-entropy (InfoNCE) loss

Design goals
  • Stay close in spirit to your ConvNetSTL: stack of Conv-BN-ReLU blocks, optional pooling indices, GAP head
  • Keep index convention 0-based for pool_after (e.g., [0, 2] => pool after blocks 0 and 2)
  • Expose forward_single() for downstream tasks (e.g., linear eval)

Notes
  • Feature dimension == encoder.out_dim, which equals the last conv width
  • Projection MLP defaults to: hidden_dim = 2 * width, proj_dim = 128
  • All outputs are L2-normalized for stability
"""
from __future__ import annotations
from typing import List, Tuple, Optional

import math
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
    """Configurable CNN encoder with optional MaxPool after specified blocks and GAP head.

    Args:
        in_ch: input channels (default 3)
        width: number of channels per conv block
        depth: number of ConvBNReLU blocks
        pool_after: list of 0-based indices; if i in this list, append MaxPool after block i
        gap: if True, global-average-pool the feature map to a vector
    """
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

    def forward_featmap(self, x: torch.Tensor) -> torch.Tensor:
        return self.features(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        if self.gap:
            x = x.mean(dim=(-2, -1))  # global average pooling
        return x

    def forward_single(self, x: torch.Tensor) -> torch.Tensor:
        """Alias for downstream usage (encodes a single batch)."""
        return self.forward(x)


class ProjectionHead(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, out_dim: int = 128):
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
        z = F.normalize(z, dim=-1)  # L2 normalize projection
        return z


class SimCLR(nn.Module):
    def __init__(self, encoder: ConvEncoder, proj_hidden: Optional[int] = None, proj_dim: int = 128):
        super().__init__()
        self.encoder = encoder
        self.proj = ProjectionHead(in_dim=encoder.out_dim, hidden_dim=proj_hidden, out_dim=proj_dim)

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Return L2-normalized encoder features (before projection)."""
        f = self.encoder(x)
        f = F.normalize(f, dim=-1)
        return f

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        f1 = self.encoder(x1)  # [B, D]
        f2 = self.encoder(x2)  # [B, D]
        z1 = self.proj(f1)     # [B, P]
        z2 = self.proj(f2)     # [B, P]
        return z1, z2


# ----------------------------
# Loss: NT-Xent (SimCLR)
# ----------------------------

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.5) -> torch.Tensor:
    """Compute the SimCLR NT-Xent loss for a batch of positive pairs (z1, z2).

    Args:
        z1, z2: [B, D] L2-normalized projection vectors for the two augmented views
        temperature: softmax temperature
    Returns:
        Scalar loss averaged over the 2B positives
    """
    assert z1.dim() == 2 and z2.dim() == 2 and z1.shape == z2.shape
    device = z1.device
    B, D = z1.shape

    z = torch.cat([z1, z2], dim=0)  # [2B, D]
    # Cosine similarity matrix [2B, 2B]
    sim = torch.matmul(z, z.t()) / temperature

    # Mask self similarities
    diag = torch.eye(2 * B, device=device, dtype=torch.bool)
    sim = sim.masked_fill(diag, float('-inf'))

    # Positives: for i in [0..B-1], pos for i is i+B; for i in [B..2B-1], pos is i-B
    pos_idx = torch.arange(B, device=device)
    pos_pairs = torch.cat([pos_idx + B, pos_idx])  # [2B]

    # Log-softmax over rows, pick positives
    log_prob = F.log_softmax(sim, dim=1)
    loss = -log_prob[torch.arange(2 * B, device=device), pos_pairs].mean()
    return loss


# ----------------------------
# Model sizing utility ("neurons")
# ----------------------------

def total_neurons(module: nn.Module) -> int:
    """Proxy size metric similar to prior code: sum of parameter counts."""
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "ConvEncoder",
    "ProjectionHead",
    "SimCLR",
    "nt_xent_loss",
    "total_neurons",
]
