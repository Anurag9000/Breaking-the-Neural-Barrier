"""
PIRL with a lightweight CNN encoder (self-supervised, PyTorch)
--------------------------------------------------------------
Idea (Misra & van der Maaten, 2020)
  • Learn representations invariant to a pretext transformation (e.g., Jigsaw).
  • Contrast clean view vs. transformed view (here: jigsaw) for same image, pushing apart others.

Components
  • ConvEncoder: configurable CNN backbone (Conv-BN-ReLU blocks, optional MaxPool, GAP)
  • ProjectionMLP: 2-layer projector head
  • Jigsaw: simple 3x3 permutation-based pretext transform
  • PIRL: symmetric contrastive loss (NT-Xent) between clean and transformed views
"""
from __future__ import annotations
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF


# ----------------------------
# Jigsaw transformation
# ----------------------------
class Jigsaw:
    DEFAULT_BANK = [
        (0,1,2,3,4,5,6,7,8), (1,0,2,3,4,5,6,7,8), (2,1,0,3,4,5,6,7,8),
        (0,2,1,3,4,5,6,7,8), (3,0,2,1,4,5,6,7,8), (6,3,0,1,4,2,7,5,8),
        (8,7,6,5,4,3,2,1,0), (0,3,6,1,4,7,2,5,8), (6,7,8,3,4,5,0,1,2)
    ]

    def __init__(self, bank: Optional[List[Tuple[int,...]]] = None, p: float = 1.0):
        self.bank = bank or self.DEFAULT_BANK
        self.p = p

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:
            return x
        C, H, W = x.shape
        h, w = H // 3, W // 3
        patches = [x[:, r*h:(r+1)*h, c*w:(c+1)*w] for r in range(3) for c in range(3)]
        perm = self.bank[torch.randint(0, len(self.bank), (1,)).item()]
        rows = [torch.cat([patches[perm[3*r + c]] for c in range(3)], dim=-1) for r in range(3)]
        return torch.cat(rows, dim=-2)


# ----------------------------
# Core CNN backbone
# ----------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class ConvEncoder(nn.Module):
    def __init__(self, in_ch=3, width=64, depth=4, pool_after: Optional[List[int]] = None, gap=True):
        super().__init__()
        pool_after = pool_after or []
        layers = []
        ch = in_ch
        for i in range(depth):
            layers.append(ConvBNReLU(ch, width))
            ch = width
            if i in set(pool_after):
                layers.append(nn.MaxPool2d(2))
        self.features = nn.Sequential(*layers)
        self.gap = gap
        self.out_dim = width
    def forward(self, x):
        x = self.features(x)
        if self.gap:
            x = x.mean(dim=(-2, -1))
        return x
    @torch.no_grad()
    def encode(self, x):
        return F.normalize(self.forward(x), dim=-1)


# ----------------------------
# Projector & PIRL main class
# ----------------------------
class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=128):
        super().__init__()
        hidden_dim = hidden_dim or (2 * in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=False)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=-1)


class PIRL(nn.Module):
    def __init__(self, encoder: ConvEncoder, proj_hidden: Optional[int] = None, proj_dim: int = 128, temperature: float = 0.2):
        super().__init__()
        self.encoder = encoder
        self.projector = ProjectionMLP(in_dim=encoder.out_dim, hidden_dim=proj_hidden, out_dim=proj_dim)
        self.tau = temperature

    def _encode(self, x):
        f = self.encoder(x)
        return self.projector(f)

    def loss(self, x, x_jig):
        z = self._encode(x)
        zt = self._encode(x_jig)
        sim = z @ zt.t() / self.tau
        labels = torch.arange(z.size(0), device=z.device)
        loss1 = F.cross_entropy(sim, labels)
        loss2 = F.cross_entropy(sim.t(), labels)
        return 0.5 * (loss1 + loss2)

    @torch.no_grad()
    def encode(self, x):
        return self.encoder.encode(x)


def total_neurons(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


__all__ = ["Jigsaw", "ConvEncoder", "ProjectionMLP", "PIRL", "total_neurons"]
