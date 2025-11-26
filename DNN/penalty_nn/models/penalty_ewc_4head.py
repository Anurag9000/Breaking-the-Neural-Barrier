"""
================================================================================
Physics-Aware Penalty DNN-EWC Model (GPU-first, 4 heads)
================================================================================

Extends the standard DNN-EWC continual-learning backbone to include a composite
loss of:
    λ₁·MSE(output) + λ₂·‖power_balance_residuals‖₂ + λ₃·‖inequality_violations‖₂

GPU-first: all tensors, buffers, and operations are initialized on the model's
CUDA device (if available) with no host↔device transfers in the hot path.
"""
from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from Dyn_DNN4OPF.models.dnn_ewc_4head import DNN_EWC_4Head
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals

class PenaltyDNN_EWC_4HEAD(DNN_EWC_4Head):
    """DNN-EWC with additional physics-aware penalty terms (4 fixed heads).

    Head mapping (task_id):
        0 → Pg   (n_gen)
        1 → Qg   (n_gen)
        2 → Va   (n_bus)
        3 → Vm   (n_bus)
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
        # Resolve device up-front (GPU-first)
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
        if v_min is None or v_max is None:
            # If not provided, create placeholders sized as number of buses
            # n_bus inferred from dims after super().__init__ when needed.
            pass
        else:
            v_min = v_min.to(dev)
            v_max = v_max.to(dev)
        # Angle limits; default to ±π if not provided
        va_min = const.get("va_min")
        va_max = const.get("va_max")
        # We may need n_bus to size defaults; infer after base init if missing.

        # Initialize base EWC (builds trunk, heads, (optional) bound layers)
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

        # If v/va limits were missing, infer sizes from internal buffers
        # n_bus and n_gen are stored in DNN_EWC as registered buffers _n_bus_tensor/_n_gen_tensor
        n_bus = int(self._n_bus_tensor.item())
        n_gen = int(self._n_gen_tensor.item())
        if v_min is None or v_max is None:
            v_min = torch.full((n_bus,), 0.0, device=dev)
            v_max = torch.full((n_bus,), 1e9, device=dev)  # effectively no clamp if absent
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

        # Configure test‑time clipping only for real bounded layers
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

        Equality residuals use the **full** state concatenation [Pg,Qg,Va,Vm]
        computed on-the-fly to avoid stale/partial signals.
        Inequality penalties are applied **head-wise** using the appropriate
        limits for Pg, Qg, Va, or Vm.
        """
        # 1) Forward current head
        y_pred = self.forward(x, task_id)

        # 2) MSE term
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # 3) Equality residual norm — build full prediction vector
        with torch.no_grad():
            pg = self.forward(x, 0)
            qg = self.forward(x, 1)
            va = self.forward(x, 2)
            vm = self.forward(x, 3)
            y_full = torch.cat([pg, qg, va, vm], dim=1)
        eq_out = power_balance_residuals(y_full, metadata or {})
        if isinstance(eq_out, (tuple, list)) and len(eq_out) == 2:
            res_r, res_i = eq_out
            eq_norm = (res_r.pow(2) + res_i.pow(2)).mean().sqrt()
        else:
            eq_norm = eq_out.pow(2).mean().sqrt()

        # 4) Inequality violations — per-head bounds
        if task_id == 0:  # Pg
            lo = self.p_min
            hi = self.p_max
        elif task_id == 1:  # Qg
            lo = self.q_min
            hi = self.q_max
        elif task_id == 2:  # Va
            lo = self.va_min
            hi = self.va_max
        elif task_id == 3:  # Vm
            lo = self.v_min
            hi = self.v_max
        else:
            raise IndexError("task_id must be in {0,1,2,3}")

        low_v  = F.relu(lo - y_pred)
        high_v = F.relu(y_pred - hi)
        ineq_norm = torch.cat([low_v, high_v], dim=1).pow(2).mean().sqrt()

        # 5) Weighted sum
        return (
            self.lambda_loss * mse
            + self.lambda_eq   * eq_norm
            + self.lambda_ineq * ineq_norm
        )


__all__ = ["PenaltyDNN_EWC"]
