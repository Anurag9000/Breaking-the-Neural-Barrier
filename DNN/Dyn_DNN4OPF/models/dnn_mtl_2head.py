"""
Fixed‑head 2‑Head MTL for OPF (Dyn_DNN4OPF)
===========================================

This variant hard‑codes **two** heads in a stable order so the `task_id`
mapping never changes across runs:

    0 → Pg_Qg  (concatenated active/reactive power per generator)
    1 → Va_Vm  (concatenated voltage angle/magnitude per bus)

Key design points
-----------------
* **GPU‑first** – all tensors/modules/bounds are created on the target CUDA
  device up‑front; no host↔device thrash inside the train/eval loops.
* **Optional bounded activation** per head via `BoundedAct`.
* Drop‑in companion for trainers that call `model(x, task_id)` and optionally
  `get_all_shared_weights()`.

Usage snippet
-------------
```python
model = DNN_MTL_2HEAD(
    input_dim=2 * n_bus,
    n_gen=n_gen,
    n_bus=n_bus,
    hidden_dim=4 * (2 * n_bus),
    use_bounds=False  # or provide 2 head‑wise bounds
).to(device)

pg_qg = model(x, task_id=0)   # (B, 2*n_gen)
va_vm = model(x, task_id=1)   # (B, 2*n_bus)
all_out = model(x)            # returns (pg_qg, va_vm)
```
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple, Sequence, List

import torch
import torch.nn as nn

try:
    # Local import – optional to allow unit‑testing without bounds
    from Dyn_DNN4OPF.utils.bounded_act import BoundedAct  # type: ignore
except ModuleNotFoundError:  # fallback if utils not on path
    BoundedAct = nn.Identity  # type: ignore

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------------------------------------------------------------------- #
#  Constants – canonical 2‑head task ids                                      #
# --------------------------------------------------------------------------- #
TASK_PG_QG: int = 0
TASK_VA_VM: int = 1
_TASK_NAMES_2H = ("Pg_Qg", "Va_Vm")


# --------------------------------------------------------------------------- #
#  Helper                                                                    #
# --------------------------------------------------------------------------- #
def _to_cuda_tensor(arr, *, dtype, device):
    """Convert anything array‑like → CUDA tensor w/ non‑blocking flag."""
    if torch.is_tensor(arr):
        return arr.to(device=device, dtype=dtype, non_blocking=True)
    return torch.as_tensor(arr, dtype=dtype, device=device)


# --------------------------------------------------------------------------- #
#  Model                                                                      #
# --------------------------------------------------------------------------- #
class DNN_MTL_2HEAD(nn.Module):
    """Fixed‑head **2‑head** MTL network with a shared fully‑connected trunk."""

    def __init__(
        self,
        *,
        input_dim: int,
        n_gen: int,
        n_bus: int,
        hidden_dim: Optional[int] = None,
        activation: nn.Module | type[nn.Module] | None = None,
        device: torch.device | None = None,
        # BoundedAct options (per‑head)
        use_bounds: bool = False,
        bounds_low: Sequence[torch.Tensor | Sequence[float]] | None = None,
        bounds_high: Sequence[torch.Tensor | Sequence[float]] | None = None,
        mask: Sequence[torch.Tensor | Sequence[bool]] | None = None,
    ) -> None:
        super().__init__()
        # GPU‑first device placement
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # Calling .to() here is harmless for an empty module; submodules are
        # also explicitly created on this device.
        self.to(self.device)

        hidden_dim = hidden_dim or 4 * input_dim

        # Accept either an activation class (e.g. nn.ReLU) or an instance
        if activation is None:
            act_cls = nn.ReLU
        elif isinstance(activation, nn.Module):
            act_cls = type(activation)
        else:
            act_cls = activation  # type: ignore[assignment]

        # Shared trunk -----------------------------------------------------
        self.trunk = nn.Sequential(
            nn.Linear(input_dim, hidden_dim, bias=True),
            act_cls(),
            nn.Linear(hidden_dim, hidden_dim, bias=True),
            act_cls(),
        ).to(self.device)

        # Two output dims: Pg+Qg and Va+Vm
        self.output_dims: List[int] = [2 * n_gen, 2 * n_bus]

        # Task heads -------------------------------------------------------
        def _head(out_dim: int) -> nn.Module:
            return nn.Linear(hidden_dim, out_dim, bias=True).to(self.device)

        self.pg_qg_head = _head(2 * n_gen)
        self.va_vm_head = _head(2 * n_bus)
        self.heads = nn.ModuleList([self.pg_qg_head, self.va_vm_head]).to(self.device)

        # (Optional) bounded clamping per head ----------------------------
        if use_bounds:
            if bounds_low is None or bounds_high is None or mask is None:
                raise ValueError("`use_bounds=True` requires length‑2 sequences for bounds_low/high and mask.")
            if not (len(bounds_low) == len(bounds_high) == len(mask) == 2):
                raise ValueError("Provide exactly 2 entries (one per head) for bounds & mask.")

            self.bound_layers = nn.ModuleList([
                BoundedAct(
                    _to_cuda_tensor(bounds_low[i], dtype=torch.float32, device=self.device),
                    _to_cuda_tensor(bounds_high[i], dtype=torch.float32, device=self.device),
                    _to_cuda_tensor(mask[i],       dtype=torch.bool,    device=self.device),
                )
                for i in range(2)
            ]).to(self.device)
        else:
            self.bound_layers = nn.ModuleList([nn.Identity(), nn.Identity()]).to(self.device)

        logger.info(
            "Initialized DNN_MTL_2HEAD: trunk %d → heads %s on %s",
            hidden_dim,
            self.output_dims,
            self.device,
        )

    # ------------------------------------------------------------------ #
    #  Forward                                                           #
    # ------------------------------------------------------------------ #
    def forward(self, x: torch.Tensor, task_id: int | None = None) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:  # type: ignore[override]
        """Forward pass for 2 heads.

        Parameters
        ----------
        x : Tensor, shape (B, input_dim)
            Input features (already on target device).
        task_id : int | None
            • *None* → return both heads as a tuple `(pg_qg, va_vm)`.
            • 0 or 1 → return only that head’s output.
        """
        x = x.to(self.device, non_blocking=True)
        shared = self.trunk(x)

        if task_id is None:
            return tuple(
                self.bound_layers[i](head(shared))
                for i, head in enumerate(self.heads)
            )  # type: ignore[return-value]

        if not (0 <= task_id < 2):
            raise ValueError(f"task_id must be 0 or 1, got {task_id}.")

        y = self.heads[task_id](shared)
        return self.bound_layers[task_id](y)

    # ------------------------------------------------------------------ #
    #  Utilities                                                         #
    # ------------------------------------------------------------------ #
    @staticmethod
    def task_name(task_id: int) -> str:
        return _TASK_NAMES_2H[task_id]

    def freeze_head(self, task_id: int) -> None:
        for p in self.heads[task_id].parameters():
            p.requires_grad_(False)

    def unfreeze_head(self, task_id: int) -> None:
        for p in self.heads[task_id].parameters():
            p.requires_grad_(True)

    def forward_all_concat(self, x: torch.Tensor) -> torch.Tensor:
        """Forward **both** heads and concatenate to (B, sum(out_dims))."""
        pg_qg, va_vm = self.forward(x)  # type: ignore[misc]
        return torch.cat((pg_qg, va_vm), dim=-1)

    def get_all_shared_weights(self):
        """Return trainable parameters (GPU‑first) for optimizers in trainers."""
        return (p for p in self.parameters() if p.requires_grad)


__all__ = [
    "DNN_MTL_2HEAD",
    "TASK_PG_QG",
    "TASK_VA_VM",
]
