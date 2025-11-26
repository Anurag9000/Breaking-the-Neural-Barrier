# penalty_dnn_ldf.py
"""Penalty replica of the LDF model (Dyn_DNN4OPF).

This variant keeps *all* baseline features (architecture, training loop,
logging, early‑stopping, test‑time clipping) *unchanged* except for the loss
function, which becomes

    L = λ1 · MSE + λ2 · ‖equality_residuals‖₂ + λ3 · ‖inequality_viols⁺‖₂

The three λ‑weights are exposed as independent hyper‑parameters
`l_loss`, `l_eq`, `l_ineq` in the config.  The model lives under the
namespace `penalty_nn.models.*` so that the vanilla codebase remains
untouched.
"""

from __future__ import annotations

import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from Dyn_DNN4OPF.utils.constraint_losses import (
    power_balance_residuals,
    mean_constraint_violation,
)
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DELTA = 1e-6  # validation‑improvement tolerance


class PenaltyLDF(nn.Module):
    """Feed‑forward OPF regressor with penalty loss."""

    # ─────────────────────────── init ────────────────────────────
    def __init__(self, config):  # noqa: D401 – simple signature
        super().__init__()

        # ── architecture dims ────────────────────────────────────
        self.config = config
        self.in_dim, self.h1_dim, self.h2_dim = config.dims
        self.n_classes = config.n_classes
        self.n_bus = self.in_dim // 2
        self.n_gen = self.n_classes // 2 - self.n_bus

        # ── penalty weights (independent!) ───────────────────────
        self.l_loss = getattr(config, "l_loss", 1.0)
        self.l_eq = getattr(config, "l_eq", 1.0)
        self.l_ineq = getattr(config, "l_ineq", 1.0)

        # ── optimisation hyper‑params ─────────────────────────────
        self.lr = config.lr
        self.patience = getattr(config, "patience", 100)
        self.delta = getattr(config, "delta", DELTA)

        # ── network layers (unchanged) ───────────────────────────
        self.fc1 = nn.Linear(self.in_dim, self.h1_dim)
        self.fc2 = nn.Linear(self.h1_dim, self.h2_dim)
        self.head = nn.Linear(self.h2_dim, self.n_classes)

    # ───────────────────────── forward ──────────────────────────
    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401 – simple forward
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

    # ─────────────────────── training loop ──────────────────────
    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader,
        constraints,
        *,
        max_epochs: int = 10000,
        delta: float = DELTA,
    ) -> float:
        """Train on *one* task with vanilla early‑stopping (pure MSE criterion)."""

        device = next(self.parameters()).device
        delta = self.delta if delta is None else delta

        optimizer, scheduler = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        best_val_mse = float("inf")
        counter = self.patience

        for epoch in range(max_epochs):
            # —— training epoch ——
            self.train()
            for xb, yb, *_ in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                preds = self(xb)

                # (1) Baseline MSE
                base_loss = F.mse_loss(preds, yb)

                # (2) Equality residuals (real + reactive power balance)
                pd, qd = xb[:, : self.n_bus], xb[:, self.n_bus : 2 * self.n_bus]
                pg = preds[:, : self.n_gen]
                qg = preds[:, self.n_gen : 2 * self.n_gen]
                va = preds[:, 2 * self.n_gen : 2 * self.n_gen + self.n_bus]
                vm = preds[:, 2 * self.n_gen + self.n_bus : 2 * (self.n_gen + self.n_bus)]

                res_P, res_Q = power_balance_residuals(
                    pg=pg,
                    qg=qg,
                    pd=pd,
                    qd=qd,
                    vm=vm,
                    va=va,
                    y_bus=constraints["eq"]["y_bus"],
                    gen_bus_idx=constraints["eq"]["gen_bus_idx"],
                    load_bus_idx=constraints["eq"]["load_bus_idx"],
                    n_bus=self.n_bus,
                )
                eq_norm = torch.sqrt((res_P.pow(2) + res_Q.pow(2)).mean())

                # (3) Inequality violations (positive part only)
                viol_P, viol_Q, viol_pg, viol_qg, viol_vm = mean_constraint_violation(
                    Y_pred=preds,
                    res_real=res_P,
                    res_imag=res_Q,
                    bounds=constraints["ineq"],
                    num_gens=self.n_gen,
                    num_buses=self.n_bus,
                )
                ineq_vec = torch.tensor(
                    [viol_P, viol_Q, viol_pg, viol_qg, viol_vm], device=device
                )
                ineq_norm = torch.norm(ineq_vec.clamp_min(0.0), p=2)

                # —— total loss ——
                loss = (
                    self.l_loss * base_loss
                    + self.l_eq * eq_norm
                    + self.l_ineq * ineq_norm
                )

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            # —— validation (pure MSE) ——
            self.eval()
            with torch.no_grad():
                val_sum = 0.0
                for xv, yv, *_ in val_loader:
                    xv, yv = xv.to(device), yv.to(device)
                    val_sum += F.mse_loss(self(xv), yv).item()
                val_mse = val_sum / len(val_loader)
            logger.info(f"[PenaltyLDF] Epoch {epoch:04d} | Val MSE: {val_mse:.6f}")

            # —— early‑stopping ——
            if val_mse < best_val_mse - delta:
                best_val_mse, counter = val_mse, self.patience
            else:
                counter -= 1
                if counter == 0:
                    logger.info("[PenaltyLDF] Early‑stopping triggered.")
                    break

        # —— test evaluation ——
        self.eval()
        with torch.no_grad():
            test_sum = 0.0
            for xt, yt in test_loader:
                xt, yt = xt.to(device), yt.to(device)
                test_sum += F.mse_loss(self(xt), yt).item()
        test_mse = test_sum / len(test_loader)
        logger.info(f"[PenaltyLDF] Final Test MSE: {test_mse:.6f}")
        return test_mse
