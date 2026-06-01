"""
BYOL with a lightweight CNN encoder (self-supervised, PyTorch)
--------------------------------------------------------------
Core idea (Grill et al., 2020)
  • Online network (encoder → projector → predictor) learns to predict target network (encoder → projector) representations
  • Target network is an exponential moving average (EMA) of the online network; stop-grad on target branch
  • No negatives, no contrastive logits

Components
  • ConvEncoder: configurable CNN backbone (Conv-BN-ReLU blocks, optional MaxPool, GAP)
  • ProjectionMLP: 2-layer MLP head
  • PredictionMLP: 2-layer MLP (usually narrower)
  • BYOL: wraps online + target nets, EMA update, symmetric loss on 2 views

Conventions
  • Pool index convention is 0-based (e.g., [0, 2] => pool after blocks 0 and 2)
  • All projection/prediction outputs are L2-normalized before cosine loss
  • total_neurons() returns parameter-count proxy
"""
from __future__ import annotations
from typing import List, Optional

import copy
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
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, out_dim: int = 256):
        super().__init__()
        h = hidden_dim or (2 * in_dim)
        self.net = nn.Sequential(
            nn.Linear(in_dim, h, bias=True),
            nn.BatchNorm1d(h),
            nn.ReLU(inplace=True),
            nn.Linear(h, out_dim, bias=False),
            nn.BatchNorm1d(out_dim),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        return F.normalize(z, dim=-1)


class PredictionMLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: Optional[int] = None, out_dim: Optional[int] = None):
        super().__init__()
        out_dim = out_dim or in_dim
        hidden_dim = hidden_dim or (in_dim // 2)
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim, bias=True),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, out_dim, bias=True),
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        p = self.net(x)
        return F.normalize(p, dim=-1)


def _ema_update(source: nn.Module, target: nn.Module, m: float):
    with torch.no_grad():
        for ps, pt in zip(source.parameters(), target.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


class BYOL(nn.Module):
    """BYOL with CNN encoder.

    online:  encoder → projector → predictor (trainable)
    target:  encoder → projector             (EMA; no grad)
    """
    def __init__(
        self,
        encoder: ConvEncoder,
        proj_hidden: Optional[int] = None,
        proj_dim: int = 256,
        pred_hidden: Optional[int] = None,
        m_ema: float = 0.996,
    ):
        super().__init__()
        self.encoder_q = encoder
        self.proj_q = ProjectionMLP(in_dim=encoder.out_dim, hidden_dim=proj_hidden, out_dim=proj_dim)
        self.pred_q = PredictionMLP(in_dim=proj_dim, hidden_dim=pred_hidden, out_dim=proj_dim)

        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.proj_k = copy.deepcopy(self.proj_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.proj_k.parameters():
            p.requires_grad = False

        self.m_ema = m_ema

    @torch.no_grad()
    def update_target(self, m: Optional[float] = None):
        m = float(self.m_ema if m is None else m)
        _ema_update(self.encoder_q, self.encoder_k, m)
        _ema_update(self.proj_q, self.proj_k, m)

    def _online(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder_q(x)
        z = self.proj_q(f)
        p = self.pred_q(z)
        return p  # normalized

    @torch.no_grad()
    def _target(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder_k(x)
        z = self.proj_k(f)
        return F.normalize(z, dim=-1)

    @staticmethod
    def _cosine_loss(p: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        # p and z are normalized; minimize negative cosine similarity
        return 2 - 2 * (p * z).sum(dim=-1).mean()

    def loss(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        # online predicts target of the other view (symmetric)
        p1 = self._online(x1)
        p2 = self._online(x2)
        with torch.no_grad():
            z1 = self._target(x1)
            z2 = self._target(x2)
        L = self._cosine_loss(p1, z2) + self._cosine_loss(p2, z1)
        return 0.5 * L

    @torch.no_grad()
    def encode(self, x: torch.Tensor, use_target: bool = True) -> torch.Tensor:
        # At evaluation time, typically use the EMA target encoder
        if use_target:
            return self.encoder_k.encode(x)
        return self.encoder_q.encode(x)


# ----------------------------
# Utility: parameter-count proxy
# ----------------------------

def total_neurons(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "ConvEncoder",
    "ProjectionMLP",
    "PredictionMLP",
    "BYOL",
    "total_neurons",
]
