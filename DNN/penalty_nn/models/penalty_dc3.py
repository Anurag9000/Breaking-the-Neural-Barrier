"""Penalty replica of Dyn_DNN4OPF.models.dnn_dc3.DNN_DC3
================================================================
Keeps every single feature of the baseline DC‑3 network (dynamic
expansion, clipping, early‑stopping, logging helpers, etc.) while
replacing the loss with the composite penalty formulation:

    L = λ1 · (baseline_loss) + λ2 · ‖h‖₂ + λ3 · ‖g⁺‖₂

where
  • baseline_loss   – the original DC‑3 loss (objective + soft penalties)
  • h               – equality residual vector  (power balance)
  • g⁺              – positive part of inequality residuals.

All λ‑weights and every other hyper‑parameter remain fully exposed and
independent of the vanilla model.  The class lives under the namespace
`penalty_nn.models` so that importing it never touches the original
code.
"""
from __future__ import annotations

from typing import Any, Optional
import torch
from Dyn_DNN4OPF.models.dnn_dc3 import DNN_DC3

__all__ = ["PenaltyDNN_DC3"]


class PenaltyDNN_DC3(DNN_DC3):
    """DC‑3 network with penalty loss as described above."""

    def __init__(
        self,
        *,
        data: Any,
        hidden_dim: Optional[int] = None,
        # Inherit **all** baseline hyper‑params ↓↓↓
        corr_steps_train: int = 50,
        corr_steps_test: int = 50,
        corr_lr: float = 1e-3,
        corr_eps: float = 1e-3,
        soft_weight: float = 1.0,
        soft_eq_frac: float = 0.5,
        use_partial: bool = False,
        use_bounds: bool = True,
        bounds_low: Optional[torch.Tensor] = None,
        bounds_high: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        # ───── new penalty coefficients ────────────────────────────
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
    ) -> None:
        # Store λs *before* baseline init so they survive `super().__init__`
        self.lambda_loss = float(lambda_loss)
        self.lambda_eq = float(lambda_eq)
        self.lambda_ineq = float(lambda_ineq)

        # Baseline initialisation (sets up net, bound layer, etc.)
        super().__init__(
            data=data,
            hidden_dim=hidden_dim,
            corr_steps_train=corr_steps_train,
            corr_steps_test=corr_steps_test,
            corr_lr=corr_lr,
            corr_eps=corr_eps,
            soft_weight=soft_weight,
            soft_eq_frac=soft_eq_frac,
            use_partial=use_partial,
            use_bounds=use_bounds,
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            mask=mask,
        )

    # ------------------------------------------------------------------
    # New penalty loss (vectorised over the batch dimension)
    # ------------------------------------------------------------------
    def loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Composite penalty loss per sample (shape = [B])."""
        # 1) baseline DC‑3 loss (objective + soft penalties)
        base = super().loss(x, y)  # [B]

        # 2) Equality residual L2 norm ‖h‖₂
        h = self.data.eq_resid(x, y)           # [B, n_eq]
        l2_eq = torch.norm(h, dim=1)           # [B]

        # 3) Positive part of inequality residuals g⁺ and its L2
        g = self.data.ineq_dist(x, y)          # [B, n_ineq]
        g_pos = torch.clamp_min(g, 0.0)
        l2_ineq = torch.norm(g_pos, dim=1)     # [B]

        # 4) Weighted sum (still returns per‑sample vector)
        return (
            self.lambda_loss * base +
            self.lambda_eq   * l2_eq +
            self.lambda_ineq * l2_ineq
        )
