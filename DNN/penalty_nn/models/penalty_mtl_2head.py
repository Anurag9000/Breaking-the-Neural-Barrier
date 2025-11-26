"""
================================================================================
Physics‑Aware Penalty Multi‑Task DNN (2‑Head MTL) for OPF
================================================================================

Extends the fixed‑head 2‑head MTL network to optimize a composite loss:
    λ₁·MSE + λ₂·‖equality_residuals‖₂ + λ₃·‖inequality_violations‖₂

GPU‑first: all parameters, buffers, and constraint constants are placed on the
CUDA device at init; no host↔device thrash in the hot path.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence, Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Dyn_DNN4OPF.models.dnn_mtl_2head import DNN_MTL_2HEAD
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals


class PenaltyDNN_MTL_2Head(DNN_MTL_2HEAD):
    """
    Fixed‑head **2‑head** MTL with physics‑aware penalty loss.

    Heads (stable order):
        0 → Pg_Qg  (2 * n_gen)
        1 → Va_Vm  (2 * n_bus)
    """

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

        # Save sizes for slicing
        self.n_gen = int(n_gen)
        self.n_bus = int(n_bus)

        # Load physical bounds and move to device (with safe defaults)
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

        # Initialize base 2‑head MTL (will call self.to(dev))
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

        # Register physics buffers on GPU (non‑persistent → lighter checkpoints)
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

    # ------------------------------- helpers ------------------------------- #
    def _full_pred(self, x: torch.Tensor) -> torch.Tensor:
        """Concatenate predictions from both heads: [Pg_Qg, Va_Vm]."""
        pg_qg, va_vm = self.forward(x)  # type: ignore[misc]
        return torch.cat((pg_qg, va_vm), dim=1)

    def _ineq_norm(self, y_pred: torch.Tensor, task_id: int) -> torch.Tensor:
        """L2 norm of inequality violations for the selected head."""
        if task_id == 0:  # Pg_Qg → split into Pg and Qg
            pg = y_pred[:, : self.n_gen]
            qg = y_pred[:, self.n_gen : 2 * self.n_gen]
            lo_pg = F.relu(self.p_min - pg)
            hi_pg = F.relu(pg - self.p_max)
            lo_qg = F.relu(self.q_min - qg)
            hi_qg = F.relu(qg - self.q_max)
            ineq = torch.cat((lo_pg, hi_pg, lo_qg, hi_qg), dim=1)
        elif task_id == 1:  # Va_Vm → split into Va and Vm
            va = y_pred[:, : self.n_bus]
            vm = y_pred[:, self.n_bus : 2 * self.n_bus]
            lo_va = F.relu(self.va_min - va)
            hi_va = F.relu(va - self.va_max)
            lo_vm = F.relu(self.v_min - vm)
            hi_vm = F.relu(vm - self.v_max)
            ineq = torch.cat((lo_va, hi_va, lo_vm, hi_vm), dim=1)
        else:
            raise ValueError(f"Invalid task_id {task_id}; expected 0 for Pg_Qg or 1 for Va_Vm.")
        return ineq.pow(2).mean().sqrt()

    # -------------------------------- loss --------------------------------- #
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
        [Pg_Qg, Va_Vm] to maintain physical coupling across heads.
        """
        # Forward the chosen head
        y_pred = self.forward(x, task_id)

        # 1) MSE
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # 2) Equality residuals norm on full current prediction
        meta = metadata or {}
        if isinstance(meta, dict):
            dev = y_pred.device
            meta = {k: (v.to(dev, non_blocking=True) if torch.is_tensor(v) else v) for k, v in meta.items()}
        y_full = self._full_pred(x)
        eq_out = power_balance_residuals(y_full, meta)
        if isinstance(eq_out, (tuple, list)) and len(eq_out) == 2:
            res_r, res_i = eq_out
            eq_norm = (res_r.pow(2) + res_i.pow(2)).mean().sqrt()
        else:
            eq_norm = eq_out.pow(2).mean().sqrt()

        # 3) Inequality norm for this head
        ineq_norm = self._ineq_norm(y_pred, task_id)

        # Weighted sum
        return (
            self.lambda_loss * mse
            + self.lambda_eq * eq_norm
            + self.lambda_ineq * ineq_norm
        )


__all__ = ["PenaltyDNN_MTL_2Head"]
