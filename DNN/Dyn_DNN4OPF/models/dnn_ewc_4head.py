"""
===============================================================================
DNN-EWC: Elastic Weight Consolidation Model Definition
===============================================================================

Fixed-head version (Pg, Qg, Va, Vm) — August 2025
-------------------------------------------------------------------------------
* Two shared ReLU layers
* Four permanent output heads:
      task_id 0 → Pg   (n_gen)
               1 → Qg   (n_gen)
               2 → Va   (n_bus)
               3 → Vm   (n_bus)
* Optional per-head BoundedAct clipping
* 100 % GPU residency after one model.to(device) call
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

import torch
import torch.nn as nn
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class DNN_EWC_4Head(nn.Module):
    """
    Feed-forward backbone used with Elastic Weight Consolidation.

    All catastrophic-forgetting countermeasures (Fisher snapshot, penalty) act
    only on the **shared** layers; the four heads are task-specific.
    """

    # ──────────────────────────────── init ──────────────────────────────────
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
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        # ── shared trunk ──
        self.shared_layers = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU()),
                nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU()),
            ]
        )

        # ── deduce bus / generator counts ──
        n_bus: int = input_dim // 2
        n_gen: int = (output_dim - 2 * n_bus) // 2
        if output_dim != 2 * n_gen + 2 * n_bus or n_gen <= 0:
            raise ValueError(
                f"Could not infer n_gen / n_bus from dims "
                f"(input={input_dim}, output={output_dim})"
            )

        # ── four fixed heads ──
        head_dims: Dict[str, int] = {
            "pg": n_gen,
            "qg": n_gen,
            "va": n_bus,
            "vm": n_bus,
        }
        self.id2name: Dict[int, str] = {idx: name for idx, name in enumerate(head_dims)}
        self.heads = nn.ModuleDict(
            {name: nn.Linear(hidden_dim, dim) for name, dim in head_dims.items()}
        )

        # ── (optional) bounded activation per head ──
        self.bound_layers = nn.ModuleDict()
        if use_bounds:
            if any(arg is None for arg in (bounds_low, bounds_high, mask)):
                raise ValueError("bounds_low, bounds_high and mask must be provided")
            # convert to tensors in case caller passed lists / numpy
            bounds_low_t = torch.as_tensor(bounds_low, dtype=torch.float32)
            bounds_high_t = torch.as_tensor(bounds_high, dtype=torch.float32)
            mask_t = torch.as_tensor(mask, dtype=torch.bool)

            if not (
                len(bounds_low_t)
                == len(bounds_high_t)
                == len(mask_t)
                == output_dim
            ):
                raise ValueError("Bounds / mask length mismatch with output_dim")

            # index slices that match the head order in downstream loader
            pg_slice = slice(0, n_gen)
            qg_slice = slice(n_gen, 2 * n_gen)
            va_slice = slice(2 * n_gen, 2 * n_gen + n_bus)
            vm_slice = slice(2 * n_gen + n_bus, 2 * n_gen + 2 * n_bus)
            name2slice = {
                "pg": pg_slice,
                "qg": qg_slice,
                "va": va_slice,
                "vm": vm_slice,
            }
            for name, sl in name2slice.items():
                self.bound_layers[name] = BoundedAct(
                    bounds_low_t[sl], bounds_high_t[sl], mask_t[sl]
                )
        else:
            self.bound_layers.update({name: nn.Identity() for name in head_dims})

        # Make sure all tensors are registered → .to(device) moves *everything*
        self.register_buffer("_n_bus_tensor", torch.tensor(n_bus))
        self.register_buffer("_n_gen_tensor", torch.tensor(n_gen))

        logger.info(
            "DNN_EWC initialised | input_dim=%d | hidden_dim=%d | n_gen=%d | n_bus=%d",
            input_dim,
            hidden_dim,
            n_gen,
            n_bus,
        )

    # ────────────────────────────── forward ─────────────────────────────────
    @torch.inference_mode(False)
    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass through the backbone and the selected head.

        Args
        ----
        x : (B, input_dim)  — feature batch on **GPU**
        task_id : int ∈ {0,1,2,3}  — selects Pg, Qg, Va, Vm head

        Returns
        -------
        (B, head_dim) prediction tensor on **GPU**
        """
        if task_id not in self.id2name:
            raise ValueError(f"task_id must be 0..3, got {task_id}")
        for layer in self.shared_layers:
            x = layer(x)
        name = self.id2name[task_id]
        x = self.heads[name](x)
        return self.bound_layers[name](x)

    # ─────────────────────────── diagnostics ────────────────────────────────
    def predict_all(self, x: torch.Tensor) -> torch.Tensor:
        """
        Convenience method — returns concatenated [Pg, Qg, Va, Vm] predictions.

        Used only for diagnostics / plotting; not in training loop.
        """
        parts = [self(x, tid) for tid in range(4)]
        return torch.cat(parts, dim=-1)

    # shared-layer weights (needed by ewc_utils.py)
    def get_all_shared_weights(self) -> List[torch.Tensor]:
        return [seq[0].weight for seq in self.shared_layers]
