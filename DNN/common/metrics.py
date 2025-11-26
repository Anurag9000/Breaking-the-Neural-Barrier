"""Metrics for OPF autoencoders."""
from __future__ import annotations

from typing import Dict, Tuple

import torch

from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals


def compute_cost(pg: torch.Tensor, coeffs: Dict[str, torch.Tensor]) -> torch.Tensor:
    cost_q = coeffs["cost_q"].to(pg.device)
    cost_l = coeffs["cost_l"].to(pg.device)
    cost_c = coeffs["cost_c"].to(pg.device)
    return (cost_q * pg.pow(2) + cost_l * pg + cost_c).sum(dim=1)


def summarise_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    pdqd: torch.Tensor,
    bounds_lo: torch.Tensor,
    bounds_hi: torch.Tensor,
    y_bus_real: torch.Tensor,
    y_bus_imag: torch.Tensor,
    gen_bus_idx: torch.Tensor,
    load_bus_idx: torch.Tensor,
    cost_coeffs: Dict[str, torch.Tensor],
) -> Dict[str, float]:
    with torch.no_grad():
        batch_size, total_dim = predictions.shape
        n_bus = pdqd.shape[1] // 2
        n_gen = total_dim // 2 - n_bus

        pg_pred = predictions[:, :n_gen]
        pg_true = targets[:, :n_gen]
        qg_pred = predictions[:, n_gen : 2 * n_gen]
        va_pred = predictions[:, 2 * n_gen : 2 * n_gen + n_bus]
        vm_pred = predictions[:, 2 * n_gen + n_bus :]

        pd = pdqd[:, :n_bus]
        qd = pdqd[:, n_bus:]
        residual_p, residual_q = power_balance_residuals(
            predictions,
            pd,
            qd,
            y_bus_real,
            y_bus_imag,
            gen_bus_idx,
            load_bus_idx,
        )
        pf_rms = torch.sqrt((residual_p.pow(2).mean() + residual_q.pow(2).mean()) / 2.0)

        below = (predictions < bounds_lo.to(predictions.device)).sum().item()
        above = (predictions > bounds_hi.to(predictions.device)).sum().item()
        n_violations = below + above

        cost_pred = compute_cost(pg_pred, cost_coeffs)
        cost_true = compute_cost(pg_true, cost_coeffs)
        cost_gap = (cost_pred - cost_true).abs().mean().item()

        overload = torch.maximum(predictions - bounds_hi.to(predictions.device), torch.zeros_like(predictions))
        underload = torch.maximum(bounds_lo.to(predictions.device) - predictions, torch.zeros_like(predictions))
        max_overload = torch.maximum(overload.max(dim=1).values, underload.max(dim=1).values).mean().item()

        vm_low = torch.maximum(bounds_lo[-n_bus:].to(predictions.device) - vm_pred, torch.zeros_like(vm_pred))
        vm_high = torch.maximum(vm_pred - bounds_hi[-n_bus:].to(predictions.device), torch.zeros_like(vm_pred))
        max_v_bounds = torch.maximum(vm_low.max(), vm_high.max()).item()

        return {
            "pf_rms": pf_rms.item(),
            "violations": float(n_violations),
            "cost_gap": cost_gap,
            "max_overload": max_overload,
            "max_v_bounds": max_v_bounds,
        }

