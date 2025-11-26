"""penalty_dnn_stl.py — Penalty variant of the STL fully‑connected network.
Keeps every feature of the baseline model while replacing the plain‑MSE loss
with a physics‑aware penalty:
    L = λ₁ · MSE + λ₂ · ||h||₂ + λ₃ · ||g⁺||₂
where
    • h  – equality (power‑balance) residuals per bus
    • g⁺ – positive part of inequality violations (Pg, Qg, Vm)
All λ‑weights, bounds, and test‑time clipping are fully configurable.
The class lives in the independent namespace `penalty_nn.models`.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
import math
from pathlib import Path
from typing import Optional

# ─── baseline network to inherit forward/expansion/clipping/etc. ─────────────
from Dyn_DNN4OPF.models.dnn_stl import FullyConnectedNet as _BaseNet

# ─── helper functions used throughout the baseline repo ─────────────────────
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds

__all__ = ["PenaltyFullyConnectedNet"]


class PenaltyFullyConnectedNet(_BaseNet):
    """STL network with physics‑aware penalty loss.

    Parameters
    ----------
    lambda_loss : float, default 1.0
        Weight λ₁ on the plain MSE term.
    lambda_eq   : float, default 1.0
        Weight λ₂ on the equality (power‑balance) residual norm.
    lambda_ineq : float, default 1.0
        Weight λ₃ on the inequality violation norm.
    case_name   : str,   default "pglib_opf_case118_ieee"
        Which OPF case to load physical bounds/topology from.  The same helper
        is used by the vanilla diagnostics, so nothing else in the codebase is
        modified.
    clip_test   : bool,  default False
        Toggle run‑time output clipping via the inherited `BoundedAct` layer.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        *,
        hidden_dim: Optional[int] = None,
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        case_name: str = "pglib_opf_case118_ieee",
        clip_test: bool = False,
        **base_kwargs,
    ) -> None:
        # capture λ‑weights first
        self.lambda_loss = float(lambda_loss)
        self.lambda_eq = float(lambda_eq)
        self.lambda_ineq = float(lambda_ineq)
        # initialise baseline (includes bounds layer, expansion hooks, etc.)
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            **base_kwargs,
        )
        # store physical constants for the chosen case (one‑off)
        const = load_case_bounds(case_name)
        self.register_buffer("p_min", const["p_min"].view(1, -1))
        self.register_buffer("p_max", const["p_max"].view(1, -1))
        self.register_buffer("q_min", const["q_min"].view(1, -1))
        self.register_buffer("q_max", const["q_max"].view(1, -1))
        self.register_buffer("v_min", const["v_min"].view(1, -1))
        self.register_buffer("v_max", const["v_max"].view(1, -1))
        # sparse Y‑bus + bus indices stay as python attributes (not tensors)
        self._y_bus = const["y_bus"]
        self._gen_bus_idx = torch.tensor(const["gen_buses"], dtype=torch.long)
        self._load_bus_idx = torch.tensor(const["load_buses"], dtype=torch.long)
        # sizes inferred from bounds
        self.n_gen = self.p_min.size(1)
        self.n_bus = self.v_min.size(1)
        # optional test‑time clipping toggle (follows baseline convention)
        self.bound_layer.apply_bounds.fill_(clip_test)  # type: ignore[attr-defined]

    # ---------------------------------------------------------------------
    # Public loss function — drop‑in replacement for nn.MSELoss in trainers
    # ---------------------------------------------------------------------
    def loss_fn(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        *,
        metadata=None,
    ) -> torch.Tensor:
        """Compute the composite penalty loss for a batch *x → y_true*.
        The signature matches the expectation of the existing trainers.
        """
        device = x.device
        # ── forward pass & plain MSE ──────────────────────────────────────
        y_pred = self(x)
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # ── equality residuals h(x) (per bus) ────────────────────────────
        # split x = [PD | QD]
        pd, qd = x[:, : self.n_bus], x[:, self.n_bus : 2 * self.n_bus]
        # split ŷ = [PG | QG | Va | Vm]
        pg = y_pred[:, : self.n_gen]
        qg = y_pred[:, self.n_gen : 2 * self.n_gen]
        va = y_pred[:, 2 * self.n_gen : 2 * self.n_gen + self.n_bus]
        vm = y_pred[:, 2 * self.n_gen + self.n_bus :]
        res_r, res_i = power_balance_residuals(
            pg,
            qg,
            pd,
            qd,
            vm,
            va,
            self._y_bus,
            self._gen_bus_idx.to(device),
            self._load_bus_idx.to(device),
            n_bus=self.n_bus,
        )
        eq_norm = torch.mean(res_r.pow(2) + res_i.pow(2))

        # ── inequality violations g⁺ (per variable) ──────────────────────
        pg_viol = torch.clamp_min(pg - self.p_max.to(device), 0) + torch.clamp_min(self.p_min.to(device) - pg, 0)
        qg_viol = torch.clamp_min(qg - self.q_max.to(device), 0) + torch.clamp_min(self.q_min.to(device) - qg, 0)
        vm_viol = torch.clamp_min(vm - self.v_max.to(device), 0) + torch.clamp_min(self.v_min.to(device) - vm, 0)
        ineq_norm = torch.mean(torch.cat([pg_viol, qg_viol, vm_viol], dim=1).pow(2))

        # ── weighted composite loss ───────────────────────────────────────
        loss = (
            self.lambda_loss * mse
            + self.lambda_eq * eq_norm
            + self.lambda_ineq * ineq_norm
        )
        return loss

    # convenience so external code can access λ‑weights cleanly
    @property
    def lambdas(self) -> tuple[float, float, float]:
        return self.lambda_loss, self.lambda_eq, self.lambda_ineq
