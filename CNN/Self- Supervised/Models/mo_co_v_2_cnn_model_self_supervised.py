"""
MoCo v2 with a lightweight CNN encoder (self-supervised, PyTorch)
------------------------------------------------------------------
Components
  • ConvEncoder: configurable CNN backbone (Conv-BN-ReLU blocks, optional MaxPool, GAP)
  • ProjectionMLP: 2-layer MLP head (as in v2)
  • MoCoV2: query (online) encoder + key (momentum) encoder with a FIFO queue of negatives

Conventions
  • Pool index convention is 0-based (e.g., [0, 2] means pool after blocks 0 and 2)
  • All embeddings used in contrastive loss are L2-normalized
  • total_neurons() returns parameter-count proxy

Loss
  • InfoNCE with one positive (current batch key) and many negatives from the queue.
    logits = [q·k_pos, q·K_neg] / T ; labels = 0
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
        )
        self.out_dim = out_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.net(x)
        z = F.normalize(z, dim=-1)
        return z


def _momentum_update(source: nn.Module, target: nn.Module, m: float):
    with torch.no_grad():
        for ps, pt in zip(source.parameters(), target.parameters()):
            pt.data.mul_(m).add_(ps.data, alpha=1.0 - m)


class MoCoV2(nn.Module):
    """Momentum Contrast v2 with CNN encoder.

    query: encoder → projector  (trainable)
    key:   encoder → projector  (EMA; no grad)
    queue: dictionary of negative keys with FIFO update
    """
    def __init__(
        self,
        encoder_q: ConvEncoder,
        proj_hidden: Optional[int] = None,
        proj_dim: int = 128,
        K: int = 16384,
        m: float = 0.99,
        T: float = 0.2,
    ):
        super().__init__()
        self.encoder_q = encoder_q
        self.proj_q = ProjectionMLP(in_dim=encoder_q.out_dim, hidden_dim=proj_hidden, out_dim=proj_dim)

        import copy
        self.encoder_k = copy.deepcopy(self.encoder_q)
        self.proj_k = copy.deepcopy(self.proj_q)
        for p in self.encoder_k.parameters():
            p.requires_grad = False
        for p in self.proj_k.parameters():
            p.requires_grad = False

        self.m = m
        self.T = T

        # Create the queue
        self.register_buffer("queue", torch.randn(proj_dim, K))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        self.K = K
        self.proj_dim = proj_dim

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys: torch.Tensor):
        # keys: [B, C] (already normalized)
        batch_size = keys.shape[0]
        ptr = int(self.queue_ptr.item())
        assert self.K % batch_size == 0, "K must be divisible by batch size for simplicity"
        self.queue[:, ptr:ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K
        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def update_momentum(self, m: Optional[float] = None):
        m = float(self.m if m is None else m)
        _momentum_update(self.encoder_q, self.encoder_k, m)
        _momentum_update(self.proj_q, self.proj_k, m)

    def _encode_q(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder_q(x)
        q = self.proj_q(f)
        return q  # L2-normalized

    @torch.no_grad()
    def _encode_k(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder_k(x)
        k = self.proj_k(f)
        return k  # L2-normalized

    def loss(self, x_q: torch.Tensor, x_k: torch.Tensor) -> torch.Tensor:
        # q: queries (online)
        q = self._encode_q(x_q)  # [B, C]
        q = F.normalize(q, dim=-1)
        # k: keys (momentum)
        with torch.no_grad():
            k = self._encode_k(x_k)
            k = F.normalize(k, dim=-1)
        # Positive logits: q·k
        l_pos = torch.sum(q * k, dim=-1, keepdim=True)  # [B,1]
        # Negative logits: q·queue
        l_neg = torch.einsum('nc,ck->nk', q, self.queue.detach())  # [B, K]
        # Logits: [B, 1+K]
        logits = torch.cat([l_pos, l_neg], dim=1) / self.T
        labels = torch.zeros(logits.size(0), dtype=torch.long, device=logits.device)
        loss = F.cross_entropy(logits, labels)
        # Update queue
        with torch.no_grad():
            self._dequeue_and_enqueue(k)
        return loss

    @torch.no_grad()
    def encode(self, x: torch.Tensor, use_key: bool = False) -> torch.Tensor:
        if use_key:
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
    "MoCoV2",
    "total_neurons",
]
