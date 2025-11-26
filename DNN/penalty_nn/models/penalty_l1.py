"""
PenaltyDNN_L1
=============
A thin wrapper around the baseline ``DNN_L1`` that augments it with a
physics‑aware penalty loss

    L = λ₁ · MSE(ŷ , y)  +  λ₂ · ‖h‖₂  +  λ₃ · ‖g⁺‖₂

where
    • h  … equality (power–balance) residuals (real *and* imaginary)
    • g⁺ … positive part of inequality constraint violations for
            Pg, Qg and |V|.

The class preserves **all** behaviour of :class:`DNN_L1` (dynamic hidden‑size
expansion, optional BoundedAct clipping, etc.) and *only* adds:

    • Three trainable/CLI‑exposed penalty weights (λ₁, λ₂, λ₃)
    • A ``loss_fn(x, y_true, metadata=None)`` method used by the
      penalty‑aware trainers (or the new run‑script).

The helper functions are re‑used verbatim from ``Dyn_DNN4OPF.utils`` – nothing
in the vanilla codebase is modified.
"""
from __future__ import annotations

from typing import Optional, Dict, Any

import torch
import torch.nn.functional as F
from torch import Tensor

from Dyn_DNN4OPF.models.dnn_l1 import DNN_L1  # baseline network
from Dyn_DNN4OPF.utils.constraint_losses import (
    power_balance_residuals as _power_balance_resid,
)

__all__ = ["PenaltyDNN_L1"]


def _violation(v: Tensor, lo: Tensor, hi: Tensor) -> Tensor:  # g⁺ helper
    """Element‑wise distance **outside** the closed interval [lo, hi]."""
    return torch.clamp(v - hi, min=0) + torch.clamp(lo - v, min=0)


class PenaltyDNN_L1(DNN_L1):
    """Drop‑in replacement for :class:`DNN_L1` with a penalty loss."""

    # ---------------------------------------------------------------------
    # Init – identical positional args + three extra penalty weights and the
    # grid constants needed to evaluate constraints.
    # ---------------------------------------------------------------------
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        *,
        use_bounds: bool = False,
        bounds_low: Optional[Tensor] = None,
        bounds_high: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        # –– penalty hyper‑parameters ––––––––––––––––––––––––––––––––––––
        lambda_loss: float = 1.0,
        lambda_eq:   float = 1.0,
        lambda_ineq: float = 1.0,
        # –– grid constants for constraint residuals ––––––––––––––––––––
        case_bounds: Optional[Dict[str, Tensor]] = None,
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

        # penalty coefficients (λᵢ)
        self.lambda_loss = lambda_loss
        self.lambda_eq   = lambda_eq
        self.lambda_ineq = lambda_ineq

        # register *all* bounds & grid tensors as buffers so they move to cuda
        self._register_case_bounds(case_bounds or {})

    # ------------------------------------------------------------------
    # Public loss used by the run‑script / trainers
    # ------------------------------------------------------------------
    def loss_fn(
        self,
        x: Tensor,                 # [B, 2·n_bus]  – PD | QD
        y_true: Tensor,            # [B, 2·(n_gen+n_bus)]  – targets
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Tensor:
        """Full penalty loss computed *inside* the network."""
        y_pred = self(x)

        # 1) baseline pure‑MSE
        base = F.mse_loss(y_pred, y_true)

        # 2) Equality residuals  (ΔP, ΔQ per bus) – L2 mean over batch
        eq_r, eq_i = _power_balance_resid(
            *self._split_pred(y_pred),
            pd=x[:, : self.n_bus],
            qd=x[:, self.n_bus : 2 * self.n_bus],
            y_bus=self.y_bus,
            gen_bus_idx=self.gen_bus_idx,
            load_bus_idx=self.load_bus_idx,
            n_bus=self.n_bus,
        )
        eq_term = torch.mean(eq_r.pow(2) + eq_i.pow(2))  # ‖h‖₂² / B

        # 3) Inequality positive violations (Pg, Qg, |V|)
        v_pg = _violation(
            y_pred[:, : self.n_gen], self.p_min, self.p_max
        )
        v_qg = _violation(
            y_pred[:, self.n_gen : 2 * self.n_gen], self.q_min, self.q_max
        )
        v_vm = _violation(
            y_pred[:, 2 * self.n_gen + self.n_bus : 2 * self.n_gen + 2 * self.n_bus],
            self.v_min,
            self.v_max,
        )
        ineq_term = torch.mean(torch.cat([v_pg, v_qg, v_vm], dim=1).pow(2))

        # — final combined loss —
        return (
            self.lambda_loss * base
            + self.lambda_eq * eq_term
            + self.lambda_ineq * ineq_term
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _register_case_bounds(self, cb: Dict[str, Tensor]) -> None:
        """Stores grid constants as *buffers* for automatic .to(device)."""
        # essential scalars
        self.n_gen = cb.get("p_min", torch.empty(0)).numel()
        self.n_bus = cb.get("v_min", torch.empty(0)).numel()

        # inequality bounds
        for key in ("p_min", "p_max", "q_min", "q_max", "v_min", "v_max"):
            val = cb.get(key)
            if val is not None:
                self.register_buffer(key, val.clone())

        # equality helpers (Y‑bus & indices) – kept as plain attrs
        self.y_bus       = cb.get("y_bus")
        self.gen_bus_idx = cb.get("gen_buses")
        self.load_bus_idx= cb.get("load_buses")

    def _split_pred(self, y: Tensor):
        """Returns (Pg, Qg, Va, Vm) tensors following canonical ordering."""
        pg = y[:, : self.n_gen]
        qg = y[:, self.n_gen : 2 * self.n_gen]
        va = y[:, 2 * self.n_gen : 2 * self.n_gen + self.n_bus]
        vm = y[:, 2 * self.n_gen + self.n_bus :]
        return pg, qg, va, vm
