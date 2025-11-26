"""
================================================================================
Physics‑Aware Penalty Multi‑Task DNN (MTL) for OPF
================================================================================

Extends the fixed‑head MTL network to optimize a composite loss:
    λ₁·MSE + λ₂·‖equality_residuals‖₂ + λ₃·‖inequality_violations‖₂

All parameters, buffers, and constraint constants are placed on GPU at init.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from Dyn_DNN4OPF.models.dnn_mtl_4head import DNN_MTL_4HEAD
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals


class PenaltyDNN_MTL_4HEAD(DNN_MTL_4HEAD):
    """Fixed‑head (Pg, Qg, Va, Vm) MTL with physics‑aware penalty loss (GPU‑first)."""

    def __init__(
        self,
        *,
        input_dim: int,
        n_gen: int,
        n_bus: int,
        hidden_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        device: Optional[torch.device] = None,
        # BoundedAct options (reused)
        use_bounds: bool = False,
        bounds_low: Sequence[Sequence[float]] | None = None,
        bounds_high: Sequence[Sequence[float]] | None = None,
        mask: Sequence[Sequence[bool]] | None = None,
        # Penalty weights
        lambda_loss: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
        case_name: str = "",
        clip_test: bool = False,
    ) -> None:
        # Determine device and place model on it
        dev = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load physical bounds and move to device (provide safe defaults)
        const: Dict[str, Any] = load_case_bounds(case_name) if case_name else {}
        # generator limits
        p_min = torch.as_tensor(const.get("p_min", torch.full((n_gen,), -float("inf"))), dtype=torch.float32, device=dev)
        p_max = torch.as_tensor(const.get("p_max", torch.full((n_gen,),  float("inf"))), dtype=torch.float32, device=dev)
        q_min = torch.as_tensor(const.get("q_min", torch.full((n_gen,), -float("inf"))), dtype=torch.float32, device=dev)
        q_max = torch.as_tensor(const.get("q_max", torch.full((n_gen,),  float("inf"))), dtype=torch.float32, device=dev)
        # bus limits
        v_min = torch.as_tensor(const.get("v_min", torch.full((n_bus,), -float("inf"))), dtype=torch.float32, device=dev)
        v_max = torch.as_tensor(const.get("v_max", torch.full((n_bus,),  float("inf"))), dtype=torch.float32, device=dev)
        # angle limits (±π if not provided)
        va_min = torch.as_tensor(const.get("va_min", torch.full((n_bus,), -math.pi)), dtype=torch.float32, device=dev)
        va_max = torch.as_tensor(const.get("va_max", torch.full((n_bus,),  math.pi)), dtype=torch.float32, device=dev)

        # Initialize base MTL (will call self.to(dev))
        super().__init__(
            input_dim=input_dim,
            n_gen=n_gen,
            n_bus=n_bus,
            hidden_dim=hidden_dim,
            activation=activation,
            device=dev,
            use_bounds=use_bounds,
            bounds_low=bounds_low,
            bounds_high=bounds_high,
            mask=mask,
        )

        # Store penalty coefficients
        self.lambda_loss = float(lambda_loss)
        self.lambda_eq = float(lambda_eq)
        self.lambda_ineq = float(lambda_ineq)

        # Register physics buffers on GPU (non‑persistent to keep checkpoints light)
        self.register_buffer("p_min", p_min, persistent=False)
        self.register_buffer("p_max", p_max, persistent=False)
        self.register_buffer("q_min", q_min, persistent=False)
        self.register_buffer("q_max", q_max, persistent=False)
        self.register_buffer("v_min", v_min, persistent=False)
        self.register_buffer("v_max", v_max, persistent=False)
        self.register_buffer("va_min", va_min, persistent=False)
        self.register_buffer("va_max", va_max, persistent=False)

        # Apply test‑time clipping if requested
        if hasattr(self, "bound_layers"):
            for layer in self.bound_layers:
                if hasattr(layer, "apply_bounds"):
                    layer.apply_bounds.fill_(bool(clip_test))

    def _ineq_norm(self, y_pred: torch.Tensor, task_id: int) -> torch.Tensor:
        """Compute L2 norm of inequality violations for the current head."""
        if task_id == 0:  # Pg
            lower, upper = self.p_min, self.p_max
        elif task_id == 1:  # Qg
            lower, upper = self.q_min, self.q_max
        elif task_id == 2:  # Va
            lower, upper = self.va_min, self.va_max
        elif task_id == 3:  # Vm
            lower, upper = self.v_min, self.v_max
        else:
            raise ValueError(f"Invalid task_id {task_id}; expected 0..3 for (Pg,Qg,Va,Vm)")

        # Broadcast to batch automatically
        lo_v = F.relu(lower - y_pred)
        hi_v = F.relu(y_pred - upper)
        ineq = torch.cat((lo_v, hi_v), dim=1)
        return ineq.pow(2).mean().sqrt()

    def _full_pred(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate predictions from all four heads in canonical order."""
        # self.forward(x) returns a tuple (Pg, Qg, Va, Vm)
        outs = self.forward(x)  # type: ignore[arg-type]
        return torch.cat(outs, dim=1)

    def loss_fn(
        self,
        x: torch.Tensor,
        y_true: torch.Tensor,
        task_id: int,
        *,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> torch.Tensor:
        """
        Compute λ₁·MSE + λ₂·‖equality_residuals‖₂ + λ₃·‖inequality_violations‖₂.

        Equality residuals are computed on the **full model prediction**
        [Pg, Qg, Va, Vm] to maintain physical coupling across heads.
        """
        # Forward selected head
        y_pred = self.forward(x, task_id)

        # 1) MSE term
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # 2) Equality residuals norm (use full current prediction)
        # Move metadata tensors to device (GPU-first) if any
        meta = metadata or {}
        if isinstance(meta, dict):
            dev = y_pred.device
            meta = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in meta.items()}

        y_full = self._full_pred(x)
        eq_out = power_balance_residuals(y_full, meta)
        if isinstance(eq_out, (tuple, list)) and len(eq_out) == 2:  # support (real, imag)
            res_real, res_imag = eq_out
            eq_norm = (res_real.pow(2) + res_imag.pow(2)).mean().sqrt()
        else:
            eq_norm = eq_out.pow(2).mean().sqrt()

        # 3) Inequality violations norm for this head
        ineq_norm = self._ineq_norm(y_pred, task_id)

        # Weighted sum
        return (
            self.lambda_loss * mse
            + self.lambda_eq * eq_norm
            + self.lambda_ineq * ineq_norm
        )


__all__ = ["PenaltyDNN_MTL"]
