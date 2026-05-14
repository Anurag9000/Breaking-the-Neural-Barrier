from __future__ import annotations

import random
from typing import Tuple

import torch

from utils.time_series_benchmarks import make_forda_loaders, make_forda_sequence_loaders


def augment_sequence(x: torch.Tensor, noise_std: float = 0.05, drop_prob: float = 0.1) -> Tuple[torch.Tensor, torch.Tensor]:
    x1 = x + noise_std * torch.randn_like(x)
    x2 = x + noise_std * torch.randn_like(x)
    if drop_prob > 0:
        m1 = (torch.rand_like(x1[..., :1]) > drop_prob).float()
        m2 = (torch.rand_like(x2[..., :1]) > drop_prob).float()
        x1 = x1 * m1
        x2 = x2 * m2
    return x1, x2


def shuffle_chunks(x: torch.Tensor, num_chunks: int) -> torch.Tensor:
    chunks = torch.chunk(x, num_chunks, dim=1)
    order = torch.randperm(len(chunks), device=x.device)
    return torch.cat([chunks[i] for i in order], dim=1)


def make_binary_order_batch(x: torch.Tensor, num_chunks: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(0, 2, (x.size(0),), device=x.device)
    out = x.clone()
    for i in range(x.size(0)):
        if labels[i].item() == 1:
            out[i] = shuffle_chunks(out[i], num_chunks)
    return out, labels


def make_multi_transform_batch(x: torch.Tensor, num_chunks: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(0, 3, (x.size(0),), device=x.device)
    out = x.clone()
    for i in range(x.size(0)):
        if labels[i].item() == 1:
            out[i] = torch.flip(out[i], dims=[0])
        elif labels[i].item() == 2:
            out[i] = shuffle_chunks(out[i], num_chunks)
    return out, labels


def make_forward_backward_batch(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    labels = torch.randint(0, 2, (x.size(0),), device=x.device)
    out = x.clone()
    for i in range(x.size(0)):
        if labels[i].item() == 1:
            out[i] = torch.flip(out[i], dims=[0])
    return out, labels


def make_boundary_batch(x: torch.Tensor, segments: int = 4) -> Tuple[torch.Tensor, torch.Tensor]:
    boundary = torch.zeros(x.size(0), x.size(1), device=x.device)
    step = max(1, x.size(1) // segments)
    for s in range(1, segments):
        boundary[:, min(x.size(1) - 1, s * step)] = 1.0
    return x, boundary


__all__ = [
    "augment_sequence",
    "make_binary_order_batch",
    "make_forda_loaders",
    "make_forda_sequence_loaders",
    "make_multi_transform_batch",
    "make_forward_backward_batch",
    "make_boundary_batch",
    "shuffle_chunks",
]
