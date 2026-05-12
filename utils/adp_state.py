from __future__ import annotations

from typing import Dict

import torch


def _copy_overlap(dst: torch.Tensor, src: torch.Tensor) -> torch.Tensor:
    """Copy overlapping tensor slices while preserving the destination init."""
    out = dst.clone()
    common = tuple(min(a, b) for a, b in zip(dst.shape, src.shape))
    slices = tuple(slice(0, c) for c in common)
    out[slices] = src[slices]
    return out


def merge_state_preserving_init(
    new_state: Dict[str, torch.Tensor],
    old_state: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """
    Merge an old checkpoint into a newly initialized state dict.

    Matching tensors are copied exactly from the old checkpoint. Resized tensors
    keep the new module's random initialization and only inherit overlapping
    slices from the old checkpoint.
    """
    merged: Dict[str, torch.Tensor] = {}
    for name, new_tensor in new_state.items():
        old_tensor = old_state.get(name)
        if old_tensor is None:
            merged[name] = new_tensor
            continue

        if old_tensor.shape == new_tensor.shape:
            merged[name] = old_tensor
            continue

        if old_tensor.ndim == new_tensor.ndim:
            merged[name] = _copy_overlap(new_tensor, old_tensor)
        else:
            merged[name] = new_tensor
    return merged
