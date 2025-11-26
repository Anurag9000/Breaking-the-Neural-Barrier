"""
Penalty replica of Dyn_DNN4OPF.models.dnn_pdl.DNNPDL
===================================================

This module defines **PenaltyDNNPDL**, a drop‑in replacement for the
vanilla *DNNPDL* that keeps every architectural and utility feature
(dynamic expansion, BoundedAct clipping, diagnostics hooks, etc.) but
exposes a new differentiable `loss_fn` implementing

    L = λ₁·baseline_obj + λ₂·‖h‖₂² + λ₃·‖ReLU(g)‖₂²

where
    • *baseline_obj* is the quadratic generator‑cost already used by the
      baseline model (see `pdl_constraints.objective`).
    • *h* are equality power‑balance residuals.
    • *g* are inequality residuals (signed distance to bounds).

The three weights (λ₁, λ₂, λ₃) are read from the model’s *cfg* so they
can be tuned independently of the vanilla hyper‑parameters.
"""
from __future__ import annotations

from types import SimpleNamespace
import torch
import torch.nn as nn

from Dyn_DNN4OPF.models.dnn_pdl import DNNPDL
from Dyn_DNN4OPF.utils.pdl_constraints import (
    compute_g, compute_h, objective,
)
from Dyn_DNN4OPF.utils.constraint_losses import inequality_residuals

__all__ = ["PenaltyDNNPDL"]


class PenaltyDNNPDL(DNNPDL):
    """Penalty version of *DNNPDL* with an embedded custom loss function.

    Parameters
    ----------
    cfg : types.SimpleNamespace
        The same namespace expected by *DNNPDL* **plus** three scalar
        attributes:
            • ``l_loss``  – λ₁ (weight on baseline objective)
            • ``l_eq``    – λ₂ (weight on equality residual)
            • ``l_ineq``  – λ₃ (weight on inequality violations)
    """

    # ------------------------------------------------------------------
    # Init
    # ------------------------------------------------------------------
    def __init__(self, cfg: SimpleNamespace):
        # read penalty weights early so they exist before super().__init__
        self.l_loss: float = float(getattr(cfg, "l_loss", 1.0))
        self.l_eq: float = float(getattr(cfg, "l_eq", 1.0))
        self.l_ineq: float = float(getattr(cfg, "l_ineq", 1.0))

        # normal DNNPDL construction (primal, dual, etc.)
        super().__init__(cfg)

    # ------------------------------------------------------------------
    # Penalty loss (used by PenaltyPDLTrainer)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _split(self, y: torch.Tensor):
        """Utility to split *y* into (Pg, Qg, Vm, Va) just like `_split_y`."""
        n_g = self.primal.bound_layer.hi.size(1) // 4  # generators
        n_b = self.primal.bound_layer.hi.size(1) // 4  # buses (Va, Vm)
        pg = y[:, : n_g]
        qg = y[:, n_g : 2 * n_g]
        vm = y[:, 2 * n_g : 2 * n_g + n_b]
        va = y[:, 2 * n_g + n_b :]
        return pg, qg, vm, va

    def loss_fn(
        self,
        x: torch.Tensor,
        batch=None,
        rho: float | None = None,  # kept for PenaltyPDLTrainer signature
    ) -> torch.Tensor:
        """Composite penalty loss (scalar)."""
        # forward pass (dual outputs unused)
        y_pred = self.primal(x)

        # baseline quadratic generation cost
        base_obj = objective(y_pred).mean()

        # power‑balance residuals (equality)
        h = compute_h(y_pred)  # shape [B,2]
        eq_term = h.pow(2).mean()

        # inequality positive violations
        g = compute_g(y_pred)  # shape [B,d]
        ineq_term = inequality_residuals(g).pow(2).mean()

        return (
            self.l_loss * base_obj
            + self.l_eq * eq_term
            + self.l_ineq * ineq_term
        )
