# penalty_dnn_mtl.py
"""
Penalty replica of the baseline DNN_MTL model.

• Keeps architecture (shared trunk, dynamic heads, bounded activation)
• Adds a differentiable penalty loss combining:
      L = λ₁ · MSE(ŷ, y) + λ₂ · ‖h‖₂ + λ₃ · ‖g⁺‖₂
  where:
      h  = equality residuals  (compute_h)
      g⁺ = positive part of inequality residuals  (compute_g → ReLU)

All λ‑weights and every other hyper‑parameter remain independently tunable from
run‑scripts/CLI.  The class inherits *all* dynamic‑expansion helpers of the
vanilla model.
"""
from __future__ import annotations

from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor

# ── internal imports (keep runtime path identical to baseline) ────────────────
from Dyn_DNN4OPF.utils.constraint_losses import inequality_residuals  # g⁺ = ReLU(g)
from Dyn_DNN4OPF.utils.pdl_constraints import compute_g, compute_h    # global fallbacks
from Dyn_DNN4OPF.models.dnn_mtl import DNN_MTL                        # base architecture

__all__ = ["PenaltyDNN_MTL"]


class PenaltyDNN_MTL(DNN_MTL):
    """DNN_MTL + quadratic penalty loss.

    Args
    ----
    input_dim : int
        Size of *x*.
    output_dims : int | list[int]
        One or more task output sizes (same semantics as baseline).
    lambda_loss : float, default 1.0
        λ₁ weight on the vanilla MSE term.
    lambda_eq : float, default 1.0
        λ₂ weight on equality‑constraint residuals.
    lambda_ineq : float, default 1.0
        λ₃ weight on inequality‑constraint violations.
    **kwargs : Any
        Forwarded verbatim to the base :class:`~Dyn_DNN4OPF.models.dnn_mtl.DNN_MTL`.
    """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        output_dims: Union[int, List[int]],
        *,
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        **kwargs,
    ) -> None:
        # store λ‑weights before super‑init so expansion utilities keep them
        self.lambda_loss = float(lambda_loss)
        self.lambda_eq = float(lambda_eq)
        self.lambda_ineq = float(lambda_ineq)
        super().__init__(input_dim=input_dim, output_dims=output_dims, **kwargs)

    # ───────────────────────────────────────────────────────────────────────
    #  Loss helper – can be called directly by training loops
    # -----------------------------------------------------------------------
    def loss_fn(
        self,
        x: Tensor,
        y_true: Tensor,
        meta: Optional[object] = None,
        *,
        task_id: int = 0,
    ) -> Tensor:
        """Return penalty loss for batch ``x`` / ``y_true``.

        The helper is **data‑agnostic**: if batches expose custom ``compute_g`` /
        ``compute_h`` methods (as in object‑datasets), we use those; otherwise we
        fall back to the global helpers registered via
        :func:`Dyn_DNN4OPF.utils.pdl_constraints.init_from_case`.
        """
        # forward pass (inherits bound‑layer, dynamic heads, etc.)
        y_pred = self.forward(x, task_id=task_id)

        # ── 1) baseline MSE ────────────────────────────────────────────────
        mse = F.mse_loss(y_pred, y_true)

        # ── 2) constraint residuals (object or global) ─────────────────────
        if meta is not None and hasattr(meta, "compute_g"):
            g = meta.compute_g(y_pred)  # type: ignore[attr-defined]
            h = meta.compute_h(y_pred)  # type: ignore[attr-defined]
        else:
            g = compute_g(y_pred)       # type: ignore[arg-type]
            h = compute_h(y_pred)       # type: ignore[arg-type]

        h_pen = torch.norm(h, p=2)                           # ‖h‖₂
        g_pen = torch.norm(inequality_residuals(g), p=2)     # ‖g⁺‖₂

        # ── 3) weighted sum ────────────────────────────────────────────────
        return (
            self.lambda_loss * mse +
            self.lambda_eq   * h_pen +
            self.lambda_ineq * g_pen
        )
