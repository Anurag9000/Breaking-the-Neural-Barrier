from __future__ import annotations

"""
PenaltyDNN_EWC
==============
A drop‑in *penalty* replica of ``Dyn_DNN4OPF.models.dnn_ewc.DNN_EWC``.

Loss definition
---------------
    L = λ₁·MSE(ŷ, y) + λ₂·‖h(ŷ)‖₂ + λ₃·‖g⁺(ŷ)‖₂
where
    • h  – per‑sample **equality** residuals returned by
            ``Dyn_DNN4OPF.utils.pdl_constraints.compute_h``
    • g⁺ – element‑wise positive part of the **inequality** residuals
            ``compute_g`` (same module)

All original features (dynamic expansion, clipping, early‑stopping, logging,
 diagnostics, etc.) are inherited unchanged from the parent class.  The new
 model exposes **λ₁, λ₂, λ₃** as first‑class hyper‑parameters and leaves every
 other keyword identical to the vanilla model, so existing run‑scripts only
 require a minimal patch.
"""

from typing import Optional
import torch
import torch.nn.functional as F

from Dyn_DNN4OPF.models.dnn_ewc import DNN_EWC  # base network
from Dyn_DNN4OPF.utils.pdl_constraints import compute_h, compute_g


class PenaltyDNN_EWC(DNN_EWC):
    """Penalty version of :class:`DNN_EWC`.  Drop‑in replacement."""

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        *,
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        **base_kw,
    ) -> None:
        """Create a :class:`PenaltyDNN_EWC`.

        Parameters
        ----------
        lambda_loss, lambda_eq, lambda_ineq
            Coefficients **λ₁, λ₂, λ₃** in the composite loss.
        base_kw
            Any keyword accepted by :class:`DNN_EWC` (bounds, mask, etc.).
        """
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            **base_kw,
        )

        self.lambda_loss = lambda_loss
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq

    # ────────────────────────────────────────────────────────────
    # Public API --------------------------------------------------
    # ────────────────────────────────────────────────────────────
    def loss_fn(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        *,
        metadata: Optional[dict] = None,  # kept for trainer‑compatibility
    ) -> torch.Tensor:
        """Composite **penalty** loss.

        The implementation relies on helper functions already used elsewhere
        in the code‑base, so *no* additional duplication is introduced.
        """
        y_pred = self.forward(x)

        # 1) baseline term
        base_term = F.mse_loss(y_pred, y_true)

        # 2) equality (power‑balance) term  ‖h‖₂ averaged over the batch
        h = compute_h(y_pred)                 # shape [B, 2]
        eq_term = h.norm(p=2, dim=1).mean()

        # 3) inequality term  ‖g⁺‖₂ averaged over the batch
        g = compute_g(y_pred)                 # shape [B, out_dim]
        g_pos = torch.clamp_min(g, 0.0)
        ineq_term = g_pos.norm(p=2, dim=1).mean()

        return (
            self.lambda_loss * base_term
            + self.lambda_eq * eq_term
            + self.lambda_ineq * ineq_term
        )
