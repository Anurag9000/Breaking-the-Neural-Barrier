"""
DeepCluster with a lightweight CNN encoder (self-supervised, PyTorch)
----------------------------------------------------------------------
Core idea
  • Alternate between (A) clustering features and (B) training a classifier to predict cluster assignments.

Components
  • ConvEncoder: configurable CNN backbone (Conv-BN-ReLU blocks, optional MaxPool, GAP)
  • ClusterHead: linear classifier over K clusters
  • kmeans_cosine: simple spherical k-means on L2-normalized features (PyTorch-only)

Conventions
  • Pool index convention: 0-based (e.g., [0, 2] => pool after blocks 0 and 2)
  • We L2-normalize features prior to clustering and classification
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
        f = self.forward(x)
        return F.normalize(f, dim=-1)


class ClusterHead(nn.Module):
    def __init__(self, in_dim: int, K: int):
        super().__init__()
        self.fc = nn.Linear(in_dim, K)
        self.K = K
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class DeepCluster(nn.Module):
    def __init__(self, encoder: ConvEncoder, K: int = 100):
        super().__init__()
        self.encoder = encoder
        self.head = ClusterHead(in_dim=encoder.out_dim, K=K)
        self.K = K
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        f = self.encoder(x)
        f = F.normalize(f, dim=-1)
        logits = self.head(f)
        return logits
    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder.encode(x)


# ----------------------------
# Spherical K-Means (cosine)
# ----------------------------
@torch.no_grad()
def kmeans_cosine(features: torch.Tensor, K: int, iters: int = 20, seed: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """Run spherical k-means on L2-normalized features.

    Args:
        features: [N, D] tensor, will be L2-normalized in function
        K: number of clusters
        iters: number of Lloyd iterations
    Returns:
        (assignments [N], centroids [K, D])
    """
    assert features.dim() == 2
    device = features.device
    N, D = features.shape
    x = F.normalize(features, dim=1)

    g = torch.Generator(device=device)
    g.manual_seed(seed)
    # k-means++ init on cosine: sample first centroid random, others by prob ~ distance^2
    idx0 = torch.randint(0, N, (1,), generator=g, device=device)
    centroids = x[idx0].clone()  # [1, D]
    for _ in range(1, K):
        # cosine distance (1 - cos)
        sim = x @ centroids.t()  # [N, c]
        dist = 1 - sim.max(dim=1).values  # [N]
        prob = dist / (dist.sum() + 1e-12)
        idx = torch.multinomial(prob, 1, generator=g)
        centroids = torch.cat([centroids, x[idx]], dim=0)

    for _ in range(iters):
        # Assign
        sims = x @ centroids.t()          # [N, K]
        assign = sims.argmax(dim=1)       # [N]
        # Update
        new_c = torch.zeros_like(centroids)
        for k in range(K):
            mask = assign == k
            if mask.any():
                new_c[k] = F.normalize(x[mask].mean(dim=0, keepdim=True), dim=1)
            else:
                # re-seed empty cluster to a random point
                ridx = torch.randint(0, N, (1,), generator=g, device=device)
                new_c[k] = x[ridx]
        # Convergence check (optional): break if tiny move
        shift = (1 - (new_c * centroids).sum(dim=1)).mean()
        centroids = new_c
        if shift < 1e-6:
            break

    return assign, centroids


# ----------------------------
# Utility: parameter-count proxy
# ----------------------------

def total_neurons(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


__all__ = [
    "ConvEncoder",
    "ClusterHead",
    "DeepCluster",
    "kmeans_cosine",
    "total_neurons",
]
