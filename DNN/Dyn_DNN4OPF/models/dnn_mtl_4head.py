"""
===============================================================================
Fixed-head Multi-Task DNN for OPF (Dyn_DNN4OPF)
===============================================================================

This version hard-codes four task heads in a *stable* order so that the
`task_id` mapping never changes across training runs:

    0 → Pg   (active power per generator)
    1 → Qg   (reactive power per generator)
    2 → Va   (voltage angle  per bus)
    3 → Vm   (voltage magnitude per bus)

Key design points
-----------------
* **No dynamic head spawning** – the architecture is fixed at init time.
* **GPU-first** – every tensor (parameters, buffers *and* constants turned
  into tensors) lives on the supplied CUDA `device` from the moment it is
  created; there is zero host↔device traffic during forward/backward.
* **Optional bounded activation** per head via `BoundedAct`.

Usage snippet
-------------
```python
model = DNN_MTL(
    input_dim=2 * n_bus,
    n_gen=n_gen,
    n_bus=n_bus,
    hidden_dim=4 * (2 * n_bus),
    use_bounds=False  # or provide bounds_* kwargs per head
).to(device)

pg = model(x, task_id=0)      # Pg prediction (B, n_gen)
all_out = model(x)            # returns (Pg, Qg, Va, Vm)
```
"""

from __future__ import annotations

import logging
from typing import Optional, Sequence, List

import torch
import torch.nn as nn

try:
    # Local import – keep optional to allow unit-testing without bounds
    from Dyn_DNN4OPF.utils.bounded_act import BoundedAct  # type: ignore
except ModuleNotFoundError:  # fallback if utils not on path
    BoundedAct = nn.Identity  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
#  Constants – canonical task ids                                             #
# --------------------------------------------------------------------------- #
TASK_PG: int = 0
TASK_QG: int = 1
TASK_VA: int = 2
TASK_VM: int = 3
_TASK_NAMES = ("Pg", "Qg", "Va", "Vm")


# --------------------------------------------------------------------------- #
#  Helper                                                                    #
# --------------------------------------------------------------------------- #
def _to_cuda_tensor(arr, *, dtype, device):
    """Convert anything array-like → CUDA tensor w/ non-blocking flag."""
    if torch.is_tensor(arr):
        return arr.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(arr, dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
#  Model                                                                      #
# --------------------------------------------------------------------------- #
class DNN_MTL_4HEAD(nn.Module):
    """Fixed-head MTL network (4 tasks) with a shared fully-connected trunk."""

    def __init__(
        self,
        *,
        input_dim: int,
        n_gen: int,
        n_bus: int,
        hidden_dim: Optional[int] = None,
        activation: nn.Module | type[nn.Module] | None = None,
        device: torch.device | None = None,
        # BoundedAct options (per-head)
        use_bounds: bool = False,
        bounds_low: Sequence[torch.Tensor | Sequence[float]] | None = None,
        bounds_high: Sequence[torch.Tensor | Sequence[float]] | None = None,
        mask: Sequence[torch.Tensor | Sequence[bool]] | None = None,
    ) -> None:
        super().__init__()

        # ------------------------------------------------------------------ #
        #  House-keeping                                                    #
        # ------------------------------------------------------------------ #
        self.device: torch.device = (
            device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        )
        # (Calling .to here is harmless for an empty module; we also place
        # submodules explicitly on device as we create them.)
        self.to(self.device)

        hidden_dim = hidden_dim or 4 * input_dim

        # accept either an activation *class* (e.g. nn.ReLU) or an *instance*
        if activation is None:
            act_cls = nn.ReLU
        elif isinstance(activation, nn.Module):
            act_cls = type(activation)
        else:
            act_cls = activation  # type: ignore[assignment]

        # ------------------------------------------------------------------ #
        #  Shared trunk                                                     #
        # ------------------------------------------------------------------ #
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            act_cls(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            act_cls(),
        ).to(self.device)

        # Output dimensions per task (Pg, Qg share n_gen; Va, Vm share n_bus)
        self.output_dims: List[int] = [n_gen, n_gen, n_bus, n_bus]

        # ------------------------------------------------------------------ #
        #  Task-specific heads                                              #
        # ------------------------------------------------------------------ #
        def _head(out_dim: int) -> nn.Module:
            return nn.Linear(hidden_dim, out_dim, bias=True)

        self.pg_head = _head(n_gen)
        self.qg_head = _head(n_gen)
        self.va_head = _head(n_bus)
        self.vm_head = _head(n_bus)
        self.heads = nn.ModuleList([
            self.pg_head,
            self.qg_head,
            self.va_head,
            self.vm_head,
        ]).to(self.device)

        # ------------------------------------------------------------------ #
        #  (Optional) bounded clamping per head                             #
        # ------------------------------------------------------------------ #
        if use_bounds:
            if bounds_low is None or bounds_high is None or mask is None:
                raise ValueError("`use_bounds=True` requires bounds_low/high and mask per head.")
            if not (len(bounds_low) == len(bounds_high) == len(mask) == 4):
                raise ValueError("Provide 4 tuples/lists (one for each head) for bounds & mask.")

            self.bound_layers = nn.ModuleList([
                BoundedAct(_to_cuda_tensor(lo, dtype=torch.float32, device=self.device),
                           _to_cuda_tensor(hi, dtype=torch.float32, device=self.device),
                           _to_cuda_tensor(msk, dtype=torch.bool,   device=self.device))
                for lo, hi, msk in zip(bounds_low, bounds_high, mask)
            ]).to(self.device)
        else:
            self.bound_layers = nn.ModuleList([nn.Identity() for _ in range(4)]).to(self.device)

        logger.info(
            "Initialized fixed-head DNN_MTL: trunk %s → heads %s on %s",
            hidden_dim,
            self.output_dims,
            self.device,
        )

    # --------------------------------------------------------------------- #
    #  Forward                                                              #
    # --------------------------------------------------------------------- #
    def forward(self, x: torch.Tensor, task_id: int | None = None):  # type: ignore[override]
        """Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)
            Input features (already on CUDA).
        task_id : int | None, default ``None``
            • *None* → return **all** heads as a tuple (Pg, Qg, Va, Vm).
            • 0-3   → return only that task’s output.
        """
        x = x.to(self.device, non_blocking=True)
        shared = self.trunk(x)

        if task_id is None:
            return tuple(
                bound(head(shared)) for head, bound in zip(self.heads, self.bound_layers)
            )

        if not 0 <= task_id < 4:
            raise ValueError(f"task_id must be 0-3, got {task_id}.")

        y = self.heads[task_id](shared)
        y = self.bound_layers[task_id](y)
        return y

    # --------------------------------------------------------------------- #
    #  Utilities                                                            #
    # --------------------------------------------------------------------- #
    @staticmethod
    def task_name(task_id: int) -> str:
        return _TASK_NAMES[task_id]

    def freeze_head(self, task_id: int) -> None:
        """Freeze parameters of a single task head."""
        for p in self.heads[task_id].parameters():
            p.requires_grad_(False)

    def unfreeze_head(self, task_id: int) -> None:
        for p in self.heads[task_id].parameters():
            p.requires_grad_(True)

    def forward_all_concat(self, x: torch.Tensor) -> torch.Tensor:
        """Forward **all** heads and concatenate to a single tensor (B, sum(out_dims))."""
        outs = self.forward(x)  # type: ignore[arg-type]
        return torch.cat(outs, dim=-1)

    def get_all_shared_weights(self):
        """Return trainable parameters (GPU-first) for optimizers in trainers."""
        return (p for p in self.parameters() if p.requires_grad)


__all__ = [
    "DNN_MTL",
    "TASK_PG",
    "TASK_QG",
    "TASK_VA",
    "TASK_VM",
]
