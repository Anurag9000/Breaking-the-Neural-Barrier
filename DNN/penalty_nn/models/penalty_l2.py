from __future__ import annotations

import torch
import torch.nn as nn
from Dyn_DNN4OPF.models.dnn_l2 import DNN_L2
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals
from typing import Optional, Tuple, Union


class PenaltyDNN_L2(DNN_L2):
    """Fully‑connected DNN with *penalty* loss:

        L = λ₁·MSE  +  λ₂·‖h(x)‖₂  +  λ₃·‖g⁺(x)‖₂

    keeping every feature of :class:`Dyn_DNN4OPF.models.dnn_l2.DNN_L2` ―
    dynamic expansion, clipping, early‑stopping friendliness, logging helpers, …
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: Optional[int] = None,
        *,
        # —— physical‑system tensors (needed for equality residuals) ——
        y_bus: Union[torch.Tensor, Tuple] | None = None,
        gen_bus_idx: torch.Tensor | None = None,
        load_bus_idx: torch.Tensor | None = None,
        # —— penalty weights ——
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        # —— original DNN_L2 args ——
        use_bounds: bool = False,
        bounds_low: torch.Tensor | None = None,
        bounds_high: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
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

        # ── store structural constants ────────────────────────────────────
        self.n_bus: int = input_dim // 2
        self.n_gen: int = (output_dim // 2) - self.n_bus

        # ── register bounds & mask for inequality term ────────────────────
        if bounds_low is not None and bounds_high is not None:
            self.register_buffer("_lo", bounds_low.clone().detach().view(1, -1))
            self.register_buffer("_hi", bounds_high.clone().detach().view(1, -1))
        else:
            self._lo = self._hi = None  # type: ignore
        self._mask = mask.bool() if mask is not None else torch.ones(output_dim, dtype=torch.bool)

        # ── grid tensors for equality term (optional) ─────────────────────
        self.y_bus = y_bus
        self.gen_bus_idx = gen_bus_idx
        self.load_bus_idx = load_bus_idx

        # ── λ‑weights as plain attributes (exposed via cfg) ───────────────
        self.lambda_loss = lambda_loss
        self.lambda_eq = lambda_eq
        self.lambda_ineq = lambda_ineq

        self._mse = nn.MSELoss()

    # ------------------------------------------------------------------
    #  Public loss_fn consumed by the *penalty* trainers
    # ------------------------------------------------------------------
    def loss_fn(self, x: torch.Tensor, y_true: torch.Tensor, *_, **__) -> torch.Tensor:  # noqa: D401
        """Return the composite penalty loss for a minibatch *x* → *y*."""
        y_pred: torch.Tensor = self.forward(x)

        # —— 1.   baseline MSE ——
        mse = self._mse(y_pred, y_true)

        # —— 2.   equality residuals ‖h‖₂ (power‑balance) ——
        if self.y_bus is not None:
            pd = x[:, : self.n_bus]
            qd = x[:, self.n_bus : 2 * self.n_bus]

            pg = y_pred[:, : self.n_gen]
            qg = y_pred[:, self.n_gen : 2 * self.n_gen]
            va = y_pred[:, 2 * self.n_gen : 2 * self.n_gen + self.n_bus]
            vm = y_pred[:, 2 * self.n_gen + self.n_bus :]

            res_r, res_i = power_balance_residuals(
                pg, qg, pd, qd, vm, va,
                self.y_bus, self.gen_bus_idx, self.load_bus_idx,
                n_bus=self.n_bus,
            )
            eq_term = torch.norm(torch.cat([res_r, res_i], dim=1), dim=1).mean()
        else:
            # allow training without the physics tensors (eq‑term disabled)
            eq_term = y_pred.new_zeros(1)

        # —— 3.   inequality violations g⁺ ——
        if self._lo is not None:
            lo = self._lo[:, self._mask]
            hi = self._hi[:, self._mask]
            yp = y_pred[:, self._mask]
            g_plus = torch.clamp(yp - hi, min=0.0) + torch.clamp(lo - yp, min=0.0)
            ineq_term = torch.norm(g_plus, dim=1).mean()
        else:
            ineq_term = y_pred.new_zeros(1)

        return (
            self.lambda_loss * mse
            + self.lambda_eq * eq_term
            + self.lambda_ineq * ineq_term
        )
