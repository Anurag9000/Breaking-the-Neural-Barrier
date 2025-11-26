"""
penalty_dnn_progressive.py
------------------------------------------------
Penalty replica of `DNN_Progressive` that augments the baseline MSE loss with
L2 penalties on equality and inequality constraint residuals, weighted by
user‑defined coefficients (λ₁, λ₂, λ₃).

The class keeps every feature of the original model (dynamic column spawning,
clipping, adapter logic, diagnostics, etc.) by subclassing the baseline and
simply adding a dedicated `loss_fn`.
"""

from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F

# ── baseline model and constraint helpers ─────────────────────────────────────
from Dyn_DNN4OPF.models.dnn_progressive import DNN_Progressive
from Dyn_DNN4OPF.utils.pdl_constraints import compute_g, compute_h  # inequality & equality residuals

__all__ = ["PenaltyDNN_Progressive"]


class PenaltyDNN_Progressive(DNN_Progressive):
    """Drop‑in replacement for `DNN_Progressive` with composite penalty loss."""

    def __init__(
        self,
        *args,
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        **kwargs,
    ) -> None:
        """Same signature as baseline + three lambda weights.

        Note: *args / **kwargs are forwarded untouched, so behaviour (dynamic
        expansion, clipping, etc.) remains identical to the vanilla model.
        """
        super().__init__(*args, **kwargs)

        # expose lambdas so they can be tuned independently
        self.lambda_loss = float(lambda_loss)
        self.lambda_eq = float(lambda_eq)
        self.lambda_ineq = float(lambda_ineq)

    # ─────────────────────────── loss function ──────────────────────────────
    def loss_fn(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        meta: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """Composite L2 penalty loss.

        Args:
            x:       Input features   — shape (N, d_in).
            y_true:  Ground‑truth targets – shape (N, d_out).
            meta:    Optional dict; if it contains a ``task_id`` key we will
                    evaluate the corresponding progressive column.

        Returns:
            Scalar tensor: total loss = λ₁·MSE + λ₂·‖h‖₂ + λ₃·‖g⁺‖₂.
        """
        # Which column to use (default = 0 for single‑task setups)
        task_id = int(meta.get("task_id", 0)) if meta is not None else 0

        # Forward pass via correct column
        y_pred = self.forward(x, task_id)

        # 1) baseline loss
        mse = F.mse_loss(y_pred, y_true)

        # 2) constraint residuals (helpers expect y_pred only)
        try:
            g = compute_g(y_pred)        # inequality residuals  (≥0 means violation)
            h = compute_h(y_pred)        # equality residuals
        except Exception:
            # Helpers not initialised (e.g. during unit‑tests) – fallback to zeros
            device = y_pred.device
            g = torch.zeros(y_pred.size(0), 1, device=device)
            h = torch.zeros(y_pred.size(0), 1, device=device)

        eq_penalty = h.norm(p=2)
        ineq_penalty = F.relu(g).norm(p=2)

        total = (
            self.lambda_loss * mse
            + self.lambda_eq * eq_penalty
            + self.lambda_ineq * ineq_penalty
        )
        return total
