from __future__ import annotations

"""penalty_dnn_elastic.py
A penalty‑augmented replica of `Dyn_DNN4OPF.models.dnn_elastic.DNN_Elastic`.
Keeps every single feature of the baseline (dynamic expansion, clipping,
logging, early‑stopping, etc.) but replaces the training loss with

    L = λ_loss·(baseline_loss) + λ_eq·‖h‖₂ + λ_ineq·‖g⁺‖₂

where
    • baseline_loss   – MSE between predictions and targets                    (per‑batch mean)
    • h               – equality residual vector returned by `compute_h()`    (PDL helpers)
    • g⁺              – ReLU‑clipped inequality residuals from `compute_g()`

All three λ‑weights are exposed as independent hyper‑parameters, alongside
_every_ knob offered by the vanilla model (hidden_dim, clipping mask, etc.).
The class uses the exact same bounded activation mechanism for test‑time
clipping as its parent and therefore remains fully plug‑and‑play inside the
existing training / diagnostics pipeline.
"""

from typing import Optional
import torch
import torch.nn as nn
import logging

from Dyn_DNN4OPF.models.dnn_elastic import DNN_Elastic
from Dyn_DNN4OPF.utils.pdl_constraints import compute_h, compute_g  # baseline helpers

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

__all__ = ["PenaltyDNNElastic"]


class PenaltyDNNElastic(DNN_Elastic):
    """Elastic‑Net DNN with physics‑aware penalty loss.

    Parameters
    ----------
    input_dim, output_dim, hidden_dim : int
        Network dimensions (identical to baseline).
    lambda1, lambda2 : float
        Elastic‑Net coefficients for L1/L2 penalties **inside the model** –
        retained for feature parity but not used in the total loss here.
    lambda_loss, lambda_eq, lambda_ineq : float
        Weights for baseline MSE, equality residuals and inequality violations.
    use_bounds, bounds_low, bounds_high, mask : see `DNN_Elastic`.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        # ─ Elastic‑Net penalties (kept for parity) ─
        lambda1: float = 1.0,
        lambda2: float = 1.0,
        # ─ NEW penalty weights ─
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        # ─ bounds / clipping ─
        use_bounds: bool = False,
        bounds_low: Optional[torch.Tensor] = None,
        bounds_high: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            lambda1=lambda1,
            lambda2=lambda2,
            use_bounds=use_bounds,
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            mask=mask,
        )

        # ─ store penalty weights & base loss ─
        self.lambda_loss = lambda_loss
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq
        self._base_loss_fn = nn.MSELoss()

        logger.debug(
            "Initialised PenaltyDNNElastic | λ_loss=%.3g λ_eq=%.3g λ_ineq=%.3g",
            lambda_loss,
            lambda_eq,
            lambda_ineq,
        )

    # ------------------------------------------------------------------
    # Public API expected by the Penalty‑* trainers
    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, x: torch.Tensor) -> torch.Tensor:
        """Helper identical to forward() but kept for backward‑compat."""
        return self.forward(x)

    def loss_fn(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        *,
        metadata: Optional[dict] = None,
    ) -> torch.Tensor:
        """Compute the composite penalty loss for a minibatch.

        The function intentionally ignores *metadata* – all required
        constants (Y‑bus, bounds, indices, …) must be registered once via
        `pdl_constraints.init_from_case(...)` **before** training starts.
        """
        # ─ 1. Predictions & pure baseline loss ─
        y_pred = self.forward(x)
        baseline = self._base_loss_fn(y_pred, y_true)

        # ─ 2. Equality residuals (power balance) ─
        h = compute_h(y_pred)                    # shape [N, 2]
        eq_pen = torch.norm(h, dim=1).mean()     # scalar

        # ─ 3. Inequality residuals (bounds) ─
        g = compute_g(y_pred)                    # shape [N, d]
        g_plus = torch.clamp_min(g, 0.0)
        ineq_pen = torch.norm(g_plus, dim=1).mean()

        # ─ 4. Composite loss ─
        total = (
            self.lambda_loss * baseline
            + self.lambda_eq * eq_pen
            + self.lambda_ineq * ineq_pen
        )

        logger.debug(
            "Penalty loss: base=%.4e | eq=%.4e | ineq=%.4e | total=%.4e",
            baseline.item(),
            eq_pen.item(),
            ineq_pen.item(),
            total.item(),
        )
        return total
