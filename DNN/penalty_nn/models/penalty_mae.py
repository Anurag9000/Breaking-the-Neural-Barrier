"""penalty_dnn_mae.py
====================================
Penalty replica of the baseline ``FullyConnectedNet`` (MAE variant).

It augments the original network with a custom ``loss_fn`` implementing the
composite penalty objective

    L = λ₁ · L_baseline  +  λ₂ · ‖equality_residuals‖₂  +  λ₃ · ‖inequality_viols⁺‖₂

All λ‑weights – together with every other hyper‑parameter of the vanilla model –
are exposed as independent constructor arguments so that nothing in the
baseline code path is affected.
"""
from __future__ import annotations

from typing import Optional
import torch
import torch.nn as nn

# ─────────────────────────── internal imports ─────────────────────────────
from Dyn_DNN4OPF.models.dnn_mae import FullyConnectedNet  # baseline net
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals

__all__ = [
    "PenaltyFullyConnectedNet",
]


class PenaltyFullyConnectedNet(FullyConnectedNet):
    """Fully‑connected network **with** composite Penalty loss.

    The architecture, clipping, logging hooks, etc. remain 100 % identical to
    the baseline :class:`FullyConnectedNet`. Only the additional state needed
    for the penalty terms is stored, and a new :py:meth:`loss_fn` is exposed so
    that *any* existing trainer can delegate the loss computation back to the
    model without touching its internals.
    """

    # ------------------------------------------------------------------
    # Construction ─ same signature + extra λ‑weights & grid constants
    # ------------------------------------------------------------------
    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        use_bounds: bool = False,
        bounds_low: Optional[torch.Tensor] = None,
        bounds_high: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        # —— new: penalty hyper‑params ——
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        # —— optional grid constants for equality residuals ——
        n_bus: Optional[int] = None,
        n_gen: Optional[int] = None,
        y_bus: Optional[torch.sparse.Tensor | tuple] = None,
        gen_bus_idx: Optional[torch.Tensor] = None,
        load_bus_idx: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            use_bounds=use_bounds,
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            mask=mask,
        )

        # —— store λ‑weights & baseline criterion (MAE) ——
        self.lambda_loss = lambda_loss
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq
        self._criterion = nn.L1Loss()

        # —— persist bounds for fast vectorised inequality check ——
        if bounds_low is not None and bounds_high is not None:
            self.register_buffer("lo", bounds_low.reshape(1, -1).clone())
            self.register_buffer("hi", bounds_high.reshape(1, -1).clone())
        else:  # allow training w/o bounds (λ₃ = 0)…
            self.lo = self.hi = None  # type: ignore[assignment]

        # —— grid meta for equality residuals (optional) ——
        self.n_bus = n_bus or (input_dim // 2)
        self.n_gen = n_gen or (output_dim // 2 - self.n_bus)
        self.y_bus = y_bus
        self.gen_bus_idx = gen_bus_idx
        self.load_bus_idx = load_bus_idx

    # ------------------------------------------------------------------
    # Private helpers for the two penalty terms
    # ------------------------------------------------------------------
    def _eq_residual_norm(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """‖equality_residuals‖₂ using power‑balance helper from baseline utils."""
        if None in (self.y_bus, self.gen_bus_idx, self.load_bus_idx):
            # If the case data is not provided just return 0 to avoid breaking
            # older scripts; users can set λ₂=0 in that case.
            return y.new_zeros(1)

        pd = x[:, : self.n_bus]
        qd = x[:, self.n_bus : 2 * self.n_bus]
        pg = y[:, : self.n_gen]
        qg = y[:, self.n_gen : 2 * self.n_gen]
        va = y[:, 2 * self.n_gen : 2 * self.n_gen + self.n_bus]
        vm = y[:, 2 * self.n_gen + self.n_bus :]

        res_r, res_i = power_balance_residuals(
            pg,
            qg,
            pd,
            qd,
            vm,
            va,
            self.y_bus,
            self.gen_bus_idx,
            self.load_bus_idx,
            n_bus=self.n_bus,
        )
        # mean L2 over batch & per‑bus residuals
        return (res_r.pow(2) + res_i.pow(2)).mean()

    def _ineq_violation_norm(self, y: torch.Tensor) -> torch.Tensor:
        """‖inequality_viols⁺‖₂ computed from simple bound clipping."""
        if self.lo is None or self.hi is None:
            return y.new_zeros(1)
        viol = torch.clamp(y - self.hi, min=0) + torch.clamp(self.lo - y, min=0)
        return viol.pow(2).mean()

    # ------------------------------------------------------------------
    # Public API ‑ trainer‑agnostic composite loss
    # ------------------------------------------------------------------
    def loss_fn(
        self,
        x: torch.Tensor,
        target: torch.Tensor,
        metadata: Optional[torch.Tensor] = None,
        **_,  # absorb extra kwargs from generic trainers
    ) -> torch.Tensor:
        """Compute composite Penalty loss for a *batch* (no reduction tricks)."""
        preds = self.forward(x)
        base = self._criterion(preds, target)
        eq   = self._eq_residual_norm(x, preds)
        ineq = self._ineq_violation_norm(preds)
        return self.lambda_loss * base + self.lambda_eq * eq + self.lambda_ineq * ineq
