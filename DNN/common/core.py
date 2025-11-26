"""Common autoencoder core utilities."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from Dyn_DNN4OPF.data.opf_loader import load_output_bounds, load_case_bounds
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals, mean_constraint_violation
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct


class ADPBackbone(nn.Module):
    """Simple MLP backbone with configurable depth and width."""

    def __init__(self, in_dim: int, hidden_dim: int, depth: int, out_dim: Optional[int] = None, activation: Optional[nn.Module] = None):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.depth = depth
        layers = []
        act = activation if activation is not None else nn.ReLU()
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(act)
            last = hidden_dim
        if out_dim is not None:
            layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsCorrection(nn.Module):
    """One-step differentiable AC power flow residual correction."""

    def __init__(self, case_name: str, *, mask: Optional[torch.Tensor] = None, steps: int = 1, step_size: float = 0.1):
        super().__init__()
        self.case_name = case_name
        self.steps = max(0, int(steps))
        self.step_size = step_size

        bounds_lo, bounds_hi = load_output_bounds(case_name)
        inferred_mask = mask if mask is not None else torch.ones_like(bounds_lo, dtype=torch.bool)
        self.bound_layer = BoundedAct(bounds_lo, bounds_hi, inferred_mask)
        self.bound_layer.apply_bounds.fill_(False)

        raw = load_case_bounds(case_name)
        self.register_buffer("y_bus_real", raw["y_bus"].real)
        self.register_buffer("y_bus_imag", raw["y_bus"].imag)
        self.register_buffer("gen_bus_idx", torch.tensor(raw["gen_buses"], dtype=torch.long))
        self.register_buffer("load_bus_idx", torch.tensor(raw["load_buses"], dtype=torch.long))

    def forward(self, outputs: torch.Tensor, pdqd: torch.Tensor) -> torch.Tensor:
        if self.steps == 0:
            return outputs
        z = outputs
        for _ in range(self.steps):
            grad = self._residual_gradient(z, pdqd)
            z = z - self.step_size * grad
        return z

    def _residual_gradient(self, outputs: torch.Tensor, pdqd: torch.Tensor) -> torch.Tensor:
        batch_size, total_dim = outputs.shape
        n_bus = pdqd.shape[1] // 2
        n_gen = total_dim // 2 - n_bus

        pg = outputs[:, :n_gen]
        qg = outputs[:, n_gen:2 * n_gen]
        va = outputs[:, 2 * n_gen:2 * n_gen + n_bus]
        vm = outputs[:, 2 * n_gen + n_bus:]

        pd = pdqd[:, :n_bus]
        qd = pdqd[:, n_bus:]

        residual_p, residual_q = power_balance_residuals(
            outputs,
            pd,
            qd,
            self.y_bus_real,
            self.y_bus_imag,
            self.gen_bus_idx,
            self.load_bus_idx,
        )

        # Voltage magnitude gradient surrogate
        g_vm = residual_p.abs() + residual_q.abs()
        if g_vm.shape[1] < vm.shape[1]:
            g_vm = F.pad(g_vm, (0, vm.shape[1] - g_vm.shape[1]))
        else:
            g_vm = g_vm[:, :vm.shape[1]]

        zeros_pg = torch.zeros_like(pg)
        zeros_qg = torch.zeros_like(qg)
        zeros_va = torch.zeros_like(va)

        grad = torch.cat([zeros_pg, zeros_qg, zeros_va, g_vm], dim=1)
        return grad


@dataclass
class CompositeLossConfig:
    alpha_cost: float = 1.0
    beta_pf: float = 10.0
    gamma_limits: float = 10.0
    eta_gen: float = 5.0
    robust_delta: Optional[float] = None


class CompositeLoss(nn.Module):
    def __init__(self, config: CompositeLossConfig):
        super().__init__()
        self.config = config

    def huber(self, diff: torch.Tensor, delta: float) -> torch.Tensor:
        abs_diff = diff.abs()
        quadratic = 0.5 * diff.pow(2)
        linear = delta * (abs_diff - 0.5 * delta)
        return torch.where(abs_diff <= delta, quadratic, linear)

    def mse_or_huber(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if self.config.robust_delta is None:
            return F.mse_loss(pred, target)
        return self.huber(pred - target, self.config.robust_delta).mean()

    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        pdqd: torch.Tensor,
        bounds_lo: torch.Tensor,
        bounds_hi: torch.Tensor,
        y_bus_real: torch.Tensor,
        y_bus_imag: torch.Tensor,
        gen_bus_idx: torch.Tensor,
        load_bus_idx: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        loss_recon = self.mse_or_huber(predictions, targets)

        pd = pdqd[:, : pdqd.shape[1] // 2]
        qd = pdqd[:, pdqd.shape[1] // 2 :]
        residual_p, residual_q = power_balance_residuals(
            predictions, pd, qd, y_bus_real, y_bus_imag, gen_bus_idx, load_bus_idx
        )
        pf_rms = torch.sqrt((residual_p.pow(2).mean() + residual_q.pow(2).mean()) / 2.0)
        limits = mean_constraint_violation(predictions, bounds_lo, bounds_hi)

        total = (
            loss_recon
            + self.config.beta_pf * pf_rms
            + self.config.gamma_limits * limits
        )

        return {
            "loss": total,
            "recon": loss_recon,
            "pf_rms": pf_rms,
            "limits": limits,
        }

