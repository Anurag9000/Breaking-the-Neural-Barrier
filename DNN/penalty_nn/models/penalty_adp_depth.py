# PenaltyADPDepth (with expansion patience)

"""Penalty variant of **ADP‑Depth** that augments the vanilla MSE objective with
soft penalties and adds **patience on expansion** for both depth and width.

Patience logic
--------------
• Accept an expansion IFF:  best_val < pre_exp_val − delta
• Allow up to N consecutive failed expansions before rollback
  (N = `config.get('trials_depth', 5)` for depth, `config.get('trials_width', 5)` for width).
• On failure within patience: keep the added capacity and immediately try another
  expansion of the same dimension.
• On patience exhaustion: rollback to the last **accepted** snapshot for that
  dimension and stop expanding that dimension.

The public API matches the vanilla `ADPDepth`.
"""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import Tuple, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────── project imports ────────────────────────────────────────────
from Dyn_DNN4OPF.models.adp_depth import (
    ADPDepth,  # base adaptive‑depth model
    expand_depth,
    expand_width,
    _resize_linear,
    _resize_head,
)
from Dyn_DNN4OPF.utils.constraint_losses import (
    power_balance_residuals,
    mean_constraint_violation,
)
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS

logger = logging.getLogger(__name__)

# ───────────────────────────────────────────────────────────────────────────────

def _cfg_get(cfg, key: str, default):
    """Dictionary-style get with attribute fallback."""
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


class PenaltyADPDepth(ADPDepth):
    """Adaptive‑Depth network with soft OPF constraints and expansion patience."""

    # ─────────────────────── init ────────────────────────────────────────────
    def __init__(
        self,
        config: SimpleNamespace | Dict[str, Any],
        *,
        lambda_loss: float = 1.0,
        lambda_equality: float = 1.0,
        lambda_ineq: float = 1.0,
        **unused,
    ) -> None:
        super().__init__(config)  # build underlying DEN trunk

        self.lambda_loss = float(lambda_loss)
        self.lambda_equality = float(lambda_equality)
        self.lambda_ineq = float(lambda_ineq)

        # constraint dictionaries will be injected by `fit_task`
        self.eq: Dict[str, Any] = getattr(config, "eq", {})
        self.ineq: Dict[str, Any] = getattr(config, "ineq", {})

        # patience knobs & failure counters
        self.trials_depth: int = int(_cfg_get(config, "trials_depth", 5))
        self.trials_width: int = int(_cfg_get(config, "trials_width", 5))
        self._depth_failures: int = 0
        self._width_failures: int = 0

    # ─────────────────────── loss ────────────────────────────────────────────
    def loss_fn(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """λ‑weighted composite loss (MSE + soft constraints)."""
        y_pred = self(x)
        base_mse = F.mse_loss(y_pred, y_true)

        n_bus = x.size(1) // 2
        n_gen = self.n_gen

        pd, qd = x[:, :n_bus], x[:, n_bus : 2 * n_bus]
        pg = y_pred[:, :n_gen]
        qg = y_pred[:, n_gen : 2 * n_gen]
        va = y_pred[:, 2 * n_gen : 2 * n_gen + n_bus]
        vm = y_pred[:, 2 * n_gen + n_bus : 2 * (n_gen + n_bus)]

        # —— equality residuals ——
        res_P, res_Q = power_balance_residuals(
            pg,
            qg,
            pd,
            qd,
            vm,
            va,
            y_bus=self.eq["y_bus"],
            gen_bus_idx=self.eq["gen_bus_idx"],
            load_bus_idx=self.eq["load_bus_idx"],
            n_bus=n_bus,
        )
        eq_norm = torch.norm(res_P, dim=1) + torch.norm(res_Q, dim=1)

        # —— inequality violations ——
        *_unused, ineq_mean = mean_constraint_violation(
            Y_pred=y_pred,
            res_real=res_P,
            res_imag=res_Q,
            bounds=self.ineq,
            num_gens=n_gen,
            num_buses=n_bus,
        )

        return (
            self.lambda_loss * base_mse
            + self.lambda_equality * eq_norm.mean()
            + self.lambda_ineq * ineq_mean
        )

    # ──────────────────── helpers ────────────────────────────────────────────
    @staticmethod
    def _train_penalty(
        model: "PenaltyADPDepth",
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        *,
        patience: int,
        delta: float,
        max_epochs: int,
        device: torch.device,
    ) -> float:
        """Vanilla early‑stopping on **pure MSE** while optimising penalty loss."""
        opt, sched = get_optimizer_scheduler(
            model.parameters(), lr=model.lr, **SCHEDULER_PARAMS
        )
        best_val = float("inf")
        counter = patience
        best_state = model.snapshot_state()

        mse_fn = nn.MSELoss()

        for _ in range(max_epochs):
            # —— training step ——
            model.train()
            for xb, yb, *_ in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = model.loss_fn(xb, yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            try:
                sched.step()
            except Exception:
                pass

            # —— validation (pure MSE) ——
            model.eval()
            val_sum = 0.0
            with torch.no_grad():
                for xb, yb, *_ in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_sum += mse_fn(model(xb), yb).item()
            val_mse = val_sum / len(val_loader)

            if val_mse < best_val - 1e-12:  # strict improvement for ES
                best_val = val_mse
                best_state = model.snapshot_state()
                counter = patience
            else:
                counter -= 1
                if counter == 0:
                    break

        model.restore_state(best_state)
        return best_val

    # ───────────────────── fit_task (override) ───────────────────────────────
    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        constraints: Dict[str, Any] | None = None,
        *,
        max_epochs: int = 10_000,
        delta: float | None = None,
    ) -> float:
        """Depth‑first / width‑second growth using the penalty loss + expansion patience."""

        #   inject constraint dicts for `loss_fn`
        if constraints is not None:
            self.eq = constraints["eq"]
            self.ineq = constraints["ineq"]

        device = next(self.parameters()).device
        patience = int(self.patience)
        inc = int(self.ex_k)
        dtmp = self.delta if delta is None else delta
        Δ = 0.0 if dtmp is None else float(dtmp)

        # Reset failure counters at task start
        self._depth_failures = 0
        self._width_failures = 0

        best_snapshot = self.snapshot_state()
        best_val = float("inf")
        stop_outer = False

        # ── OUTER: series of depth attempts, each followed by width sweeps ──
        while not stop_outer:
            # Depth capacity guard
            if len(self.hidden_layers) >= self.max_depth:
                logger.info("Reached max_depth — no further depth expansion.")
                break

            # Baseline for the upcoming DEPTH series (MSE-based, updated only on accept)
            pre_depth_snapshot = copy.deepcopy(self)
            pre_depth_val = best_val
            self._depth_failures = 0

            # ── DEPTH series with patience ───────────────────────────────────
            while True:
                expand_depth(self)
                d_val = self._train_penalty(
                    self,
                    train_loader,
                    val_loader,
                    patience=patience,
                    delta=Δ,
                    max_epochs=max_epochs,
                    device=device,
                )

                if d_val < pre_depth_val - Δ:
                    # ACCEPT depth → reset counter, update best and advance baseline
                    self._depth_failures = 0
                    best_val = d_val
                    best_snapshot = self.snapshot_state()
                    pre_depth_snapshot = copy.deepcopy(self)
                    pre_depth_val = best_val
                    logger.info("Depth expansion accepted; resetting failure counter.")
                    # Move on to width sweeps at this depth
                    break
                else:
                    # FAIL depth
                    self._depth_failures += 1
                    k, N = self._depth_failures, int(self.trials_depth)
                    if k < N:
                        logger.info(
                            f"No val improvement after depth expansion (trial {k}/{N}); trying another depth expansion.")
                        # keep architecture; immediately attempt another depth expansion
                        continue
                    else:
                        logger.info(
                            f"Depth expansions without improvement reached {N}; rolling back and stopping depth.")
                        # rollback to baseline before this depth series and stop training (depth-first policy)
                        self.restore_state(pre_depth_snapshot)
                        stop_outer = True
                        break

            if stop_outer:
                break

            # ── WIDTH sweeps at current depth with patience ─────────────────
            pre_width_snapshot = copy.deepcopy(self)
            pre_width_val = best_val
            self._width_failures = 0

            while True:
                # capacity guards
                total_neurons = sum(l.out_features for l in self.hidden_layers)
                curr_width = self.hidden_layers[-1].out_features
                if (
                    total_neurons + inc * len(self.hidden_layers) > self.max_neurons
                    or curr_width + inc > self.max_width
                ):
                    logger.info("Reached max_neurons / max_width — stop width expansion.")
                    break

                expand_width(self, inc)
                w_val = self._train_penalty(
                    self,
                    train_loader,
                    val_loader,
                    patience=patience,
                    delta=Δ,
                    max_epochs=max_epochs,
                    device=device,
                )

                if w_val < pre_width_val - Δ:
                    # ACCEPT width → reset counter, update best and advance baseline
                    self._width_failures = 0
                    best_val = w_val
                    best_snapshot = self.snapshot_state()
                    pre_width_snapshot = copy.deepcopy(self)
                    pre_width_val = best_val
                    logger.info("Width expansion accepted; resetting failure counter.")
                    # try another width increment (loop continues)
                    continue
                else:
                    # FAIL width
                    self._width_failures += 1
                    k, N = self._width_failures, int(self.trials_width)
                    if k < N:
                        logger.info(
                            f"No val improvement after width expansion (trial {k}/{N}); trying another width expansion.")
                        # keep widened net; attempt another width increment
                        continue
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back widths and stopping width series.")
                        # rollback to baseline before this width series and stop widths
                        self.restore_state(pre_width_snapshot)
                        break

            # After width series ends (capacity or rollback), loop back to start another
            # depth series if allowed by max_depth

        # —— final restore ——
        self.restore_state(best_snapshot)
        return best_val

    # ─────────────────── convenience ─────────────────────────────────────────
    @property
    def lambdas(self) -> Tuple[float, float, float]:
        """Return *(λ_loss, λ_eq, λ_ineq)*."""
        return self.lambda_loss, self.lambda_equality, self.lambda_ineq


__all__ = [
    "PenaltyADPDepth",
    "_resize_linear",
    "_resize_head",
    "expand_depth",
    "expand_width",
]
