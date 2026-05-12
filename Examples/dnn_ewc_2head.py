"""
===============================================================================
DNN-EWC (2-HEAD): Elastic Weight Consolidation Model Definition
===============================================================================

Two-head fixed version (Pg_Qg, Va_Vm) — August 2025
-------------------------------------------------------------------------------
* Two shared ReLU layers
* Two permanent output heads:
      task_id 0 → Pg_Qg (2 * n_gen)
               1 → Va_Vm (2 * n_bus)
* Optional per-head BoundedAct clipping
* GPU-first: pick a device once and keep all tensors/modules there
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DNN_EWC_2HEAD(nn.Module):
    """
    Feed-forward backbone used with Elastic Weight Consolidation (2 heads).

    All catastrophic-forgetting countermeasures (Fisher snapshot, penalty) act
    only on the **shared** layers; the two heads are task-specific.
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        use_bounds: bool = False,
        bounds_low: Optional[torch.Tensor | List[float]] = None,
        bounds_high: Optional[torch.Tensor | List[float]] = None,
        mask: Optional[torch.Tensor | List[int]] = None,
        device: Optional[torch.device] = None,
    ) -> None:
        super().__init__()

        # Choose device once (GPU-first) and keep it
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        # ── deduce bus / generator counts ──
        n_bus: int = input_dim // 2
        n_gen: int = (output_dim - 2 * n_bus) // 2
        if output_dim != 2 * n_gen + 2 * n_bus or n_gen <= 0:
            raise ValueError(
                f"Could not infer n_gen / n_bus from dims (input={input_dim}, output={output_dim})"
            )

        # ── shared trunk ──
        self.shared_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()),
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()),
            ]
        ).to(self.device)

        # ── two fixed heads ──
        self.id2name: Dict[int, str] = {0: "pg_qg", 1: "va_vm"}
        self.heads = nn.ModuleDict({
            "pg_qg": nn.Linear(hidden_dim, 2 * n_gen),
            "va_vm": nn.Linear(hidden_dim, 2 * n_bus),
        }).to(self.device)

        # ── (optional) bounded activation per head ──
        self.bound_layers = nn.ModuleDict()
        if use_bounds:
            if any(arg is None for arg in (bounds_low, bounds_high, mask)):
                raise ValueError("bounds_low, bounds_high and mask must be provided when use_bounds=True")
            # convert to tensors in case caller passed lists / numpy
            bounds_low_t = torch.as_tensor(bounds_low, dtype=torch.float64, device=self.device)
            bounds_high_t = torch.as_tensor(bounds_high, dtype=torch.float64, device=self.device)
            mask_t = torch.as_tensor(mask, dtype=torch.bool, device=self.device)

            if not (
                len(bounds_low_t) == len(bounds_high_t) == len(mask_t) == output_dim
            ):
                raise ValueError("Bounds / mask length mismatch with output_dim")

            # slices that match the concatenated output ordering [Pg,Qg,Va,Vm]
            sl_pgqg = slice(0, 2 * n_gen)
            sl_vavm = slice(2 * n_gen, 2 * n_gen + 2 * n_bus)

            self.bound_layers["pg_qg"] = BoundedAct(
                bounds_low_t[sl_pgqg], bounds_high_t[sl_pgqg], mask_t[sl_pgqg]
            )
            self.bound_layers["va_vm"] = BoundedAct(
                bounds_low_t[sl_vavm], bounds_high_t[sl_vavm], mask_t[sl_vavm]
            )
        else:
            self.bound_layers["pg_qg"] = nn.Identity()
            self.bound_layers["va_vm"] = nn.Identity()
        self.bound_layers.to(self.device)

        # Register constants for external use (move with .to(device))
        self.register_buffer("_n_gen", torch.tensor(n_gen, device=self.device))
        self.register_buffer("_n_bus", torch.tensor(n_bus, device=self.device))

        logger.info(
            "DNN_EWC_2HEAD initialised | input_dim=%d | hidden_dim=%d | n_gen=%d | n_bus=%d",
            input_dim,
            hidden_dim,
            n_gen,
            n_bus,
        )

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass through shared trunk and selected head.
        task_id: 0 for pg_qg, 1 for va_vm.
        Returns: (B, head_dim) prediction on the module's device.
        """
        if task_id not in self.id2name:
            raise ValueError(f"task_id must be 0 or 1, got {task_id}")

        x = x.to(self.device, non_blocking=True)
        for layer in self.shared_layers:
            x = layer(x)
        name = self.id2name[task_id]
        out = self.heads[name](x)
        return self.bound_layers[name](out)

    def predict_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Returns concatenated [Pg+Qg, Va+Vm] predictions for diagnostics.
        """
        parts = [self(x, tid) for tid in (0, 1)]
        return torch.cat(parts, dim=-1)

    # shared-layer weights (needed by ewc_utils.py)
    def get_all_shared_weights(self) -> List[torch.Tensor]:
        return [seq[0].weight for seq in self.shared_layers]


__all__ = ["DNN_EWC_2HEAD"]
