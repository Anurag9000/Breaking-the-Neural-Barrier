"""
================================================================================
Physics-Aware Penalty DNN-EWC Model (GPU-first, 2 heads)
================================================================================

Extends the 2-head EWC backbone to optimize a composite loss:
    λ₁·MSE + λ₂·‖power_balance_residuals‖₂ + λ₃·‖inequality_violations‖₂

Heads (task_id):
    0 → Pg_Qg  (2*n_gen)
    1 → Va_Vm  (2*n_bus)

GPU-first: all constants/buffers are created on the model's device; no host↔device
transfers in the hot path.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Dyn_DNN4OPF.models.dnn_ewc_2head import DNN_EWC_2HEAD
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals


class PenaltyDNN_EWC_2Head(DNN_EWC_2HEAD):
    """EWC (2-head) with physics-aware penalty terms.

    task_id mapping:
        0 → Pg_Qg  (inequalities vs p/q limits)
        1 → Va_Vm  (inequalities vs va/vm limits)
    """

    def __init__(
        self,
        *,
        input_dim: int,
        output_dim: int,
        hidden_dim: int | None = None,
        use_bounds: bool = False,
        bounds_low: torch.Tensor | list[float] | None = None,
        bounds_high: torch.Tensor | list[float] | None = None,
        mask: torch.Tensor | list[int] | None = None,
        lambda_loss: float,
        lambda_eq: float,
        lambda_ineq: float,
        case_name: str,
        clip_test: bool = False,
        device: torch.device | str | None = None,
    ) -> None:
        # Resolve device (GPU-first)
        dev = (
            torch.device(device) if isinstance(device, str)
            else (device if isinstance(device, torch.device) else torch.device("cuda" if torch.cuda.is_available() else "cpu"))
        )

        # Load physics bounds and move to device
        const = load_case_bounds(case_name)
        p_min = const["p_min"].to(dev)
        p_max = const["p_max"].to(dev)
        q_min = const.get("q_min", torch.empty_like(p_min)).to(dev)
        q_max = const.get("q_max", torch.empty_like(p_min)).to(dev)
        v_min = const.get("v_min")
        v_max = const.get("v_max")
        va_min = const.get("va_min")
        va_max = const.get("va_max")

        # Initialize base EWC 2-head (builds trunk, heads, optional bound layers)
        super().__init__(
            input_dim=input_dim,
            output_dim=output_dim,
            hidden_dim=hidden_dim,
            use_bounds=use_bounds,
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            mask=mask,
        )
        self.to(dev)

        # Infer sizes for defaults if some bounds are missing
        n_bus = int(self._n_bus.item())  # registered in DNN_EWC_2HEAD
        n_gen = int(self._n_gen.item())
        if v_min is None or v_max is None:
            v_min = torch.full((n_bus,), 0.0, device=dev)
            v_max = torch.full((n_bus,), 1e9, device=dev)
        else:
            v_min = v_min.to(dev)
            v_max = v_max.to(dev)
        if va_min is None or va_max is None:
            va_min = torch.full((n_bus,), -math.pi, device=dev)
            va_max = torch.full((n_bus,),  math.pi, device=dev)
        else:
            va_min = va_min.to(dev)
            va_max = va_max.to(dev)

        # Store penalty coefficients
        self.lambda_loss = float(lambda_loss)
        self.lambda_eq   = float(lambda_eq)
        self.lambda_ineq = float(lambda_ineq)

        # Register physics constants as non-trainable buffers on device
        self.register_buffer("p_min",  p_min,  persistent=False)
        self.register_buffer("p_max",  p_max,  persistent=False)
        self.register_buffer("q_min",  q_min,  persistent=False)
        self.register_buffer("q_max",  q_max,  persistent=False)
        self.register_buffer("v_min",  v_min,  persistent=False)
        self.register_buffer("v_max",  v_max,  persistent=False)
        self.register_buffer("va_min", va_min, persistent=False)
        self.register_buffer("va_max", va_max, persistent=False)

        # Configure test-time clipping safely (only if BoundedAct present)
        for _, layer in self.bound_layers.items():
            if hasattr(layer, "apply_bounds"):
                layer.apply_bounds.fill_(bool(clip_test))

    # ------------------------------------------------------------------ #
    #  Loss                                                              #
    # ------------------------------------------------------------------ #
    def loss_fn(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        task_id: int,
        *,
        metadata=None,
    ) -> torch.Tensor:
        """Composite physics-aware loss for the selected head.

        Equality term uses the **full** prediction [Pg,Qg,Va,Vm] assembled from
        both heads (no grad through the non-active head to avoid leakage).
        Inequality penalties are applied per head with appropriate limits.
        """
        # 1) Forward current head
        y_pred = self.forward(x, task_id)

        # 2) MSE
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # 3) Equality residuals on full state
        with torch.no_grad():
            y_pgqg = self.forward(x, 0)
            y_vavm = self.forward(x, 1)
            y_full = torch.cat([y_pgqg, y_vavm], dim=1)
        eq_out = power_balance_residuals(y_full, metadata or {})
        if isinstance(eq_out, (tuple, list)) and len(eq_out) == 2:
            r, i = eq_out
            eq_norm = (r.pow(2) + i.pow(2)).mean().sqrt()
        else:
            eq_norm = eq_out.pow(2).mean().sqrt()

        # 4) Inequality violations per head
        if task_id == 0:  # Pg_Qg
            lower = torch.cat([self.p_min, self.q_min])
            upper = torch.cat([self.p_max, self.q_max])
        elif task_id == 1:  # Va_Vm
            lower = torch.cat([self.va_min, self.v_min])
            upper = torch.cat([self.va_max, self.v_max])
        else:
            raise IndexError("task_id must be 0 (Pg_Qg) or 1 (Va_Vm)")

        lo_v  = F.relu(lower - y_pred)
        hi_v  = F.relu(y_pred - upper)
        ineq_norm = torch.cat([lo_v, hi_v], dim=1).pow(2).mean().sqrt()

        # 5) Weighted sum
        return (
            self.lambda_loss * mse
            + self.lambda_eq   * eq_norm
            + self.lambda_ineq * ineq_norm
        )


__all__ = ["PenaltyDNN_EWC_2Head"]
