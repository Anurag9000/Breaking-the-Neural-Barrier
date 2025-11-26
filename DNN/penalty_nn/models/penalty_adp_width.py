# penalty_adp_width.py (with expansion patience)
"""Penalty variant of **ADP‑Width** with *soft* OPF penalties **and**
patience on expansion for both width and depth.

Patience logic
--------------
• Accept an expansion IFF:  best_val < pre_exp_val − delta
• Allow up to N consecutive failed expansions before rollback
  (N = `config.get('trials_width', 5)` for width, `config.get('trials_depth', 5)` for depth)
• On failure within patience: keep the added capacity and immediately try
  another expansion of the same dimension.
• On patience exhaustion: rollback to the last *accepted* snapshot for that
  dimension and stop expanding that dimension.

Public API remains drop‑in compatible with the vanilla `ADPWidth`.
"""
from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from typing import Dict, Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ────────────────── project imports ────────────────────────────────────────────
from Dyn_DNN4OPF.models.adp_width import (
    ADPWidth,
    expand_width,
    expand_depth,
    _resize_linear,
    _resize_head,
)
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals, mean_constraint_violation
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

# ═══════════════════════════════════════════════════════════════════════════════
#  Model definition
# ═══════════════════════════════════════════════════════════════════════════════

class PenaltyADPWidth(ADPWidth):
    """Adaptive **width‑first** network with soft OPF penalties and expansion patience."""

    # ───────────────────────────── init ──────────────────────────────────────
    def __init__(
        self,
        config: SimpleNamespace | Dict[str, Any],
        *,
        lambda_loss: float = 1.0,
        lambda_equality: float = 1.0,
        lambda_ineq: float = 1.0,
        **unused,
    ) -> None:
        super().__init__(config)  # build base ADP‑Width trunk

        # penalty coefficients ------------------------------------------------
        self.lambda_loss = float(lambda_loss)
        self.lambda_equality = float(lambda_equality)
        self.lambda_ineq = float(lambda_ineq)

        # constraint dictionaries injected by `fit_task`
        self.eq: Dict[str, Any] = getattr(config, "eq", {})
        self.ineq: Dict[str, Any] = getattr(config, "ineq", {})

        # patience knobs & failure counters
        self.trials_width: int = int(_cfg_get(config, "trials_width", 5))
        self.trials_depth: int = int(_cfg_get(config, "trials_depth", 5))
        self._width_failures: int = 0
        self._depth_failures: int = 0

    # ─────────────────────────── loss fn ─────────────────────────────────────
    def loss_fn(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        """Composite loss: *pure* MSE + λ‑weighted equality & inequality terms."""
        y_pred = self(x)
        base_mse = F.mse_loss(y_pred, y_true)

        n_bus = x.size(1) // 2
        n_gen = self.n_gen

        # unpack --------------------------------------------------------------
        pd, qd = x[:, :n_bus], x[:, n_bus : 2 * n_bus]
        pg = y_pred[:, :n_gen]
        qg = y_pred[:, n_gen : 2 * n_gen]
        va = y_pred[:, 2 * n_gen : 2 * n_gen + n_bus]
        vm = y_pred[:, 2 * n_gen + n_bus : 2 * (n_gen + n_bus)]

        # equality residuals --------------------------------------------------
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

        # inequality violations ----------------------------------------------
        *_u, ineq_mean = mean_constraint_violation(
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

    # ──────────────────────── helpers ───────────────────────────────────────
    @staticmethod
    def _train_penalty(
        model: "PenaltyADPWidth",
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
        *,
        patience: int,
        delta: float,
        max_epochs: int,
        device: torch.device,
    ) -> float:
        """Early‑stopping **on pure MSE** while optimising the penalty loss."""
        opt, sch = get_optimizer_scheduler(
            model.parameters(), lr=model.lr, **SCHEDULER_PARAMS
        )
        best_val = float("inf")
        counter = patience
        best_state = model.snapshot_state()
        mse_fn = nn.MSELoss()

        for _ in range(max_epochs):
            # —— train ——
            model.train()
            for xb, yb, *_ in train_loader:
                xb, yb = xb.to(device), yb.to(device)
                loss = model.loss_fn(xb, yb)
                opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
            try:
                sch.step()
            except Exception:
                pass

            # —— validate ——
            model.eval(); val_sum = 0.0
            with torch.no_grad():
                for xb, yb, *_ in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_sum += mse_fn(model(xb), yb).item()
            val_mse = val_sum / len(val_loader)

            if val_mse < best_val - 1e-12:  # strict ES improvement
                best_val = val_mse; best_state = model.snapshot_state(); counter = patience
            else:
                counter -= 1
                if counter == 0:
                    break

        model.restore_state(best_state)
        return best_val

    # ───────────────────────── fit_task ─────────────────────────────────────
    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        constraints: Dict[str, Any] | None = None,
        *,
        max_epochs: int = 10_000,
        delta: float | None = None,
    ) -> float:  # type: ignore[override]
        """Width‑first / depth‑second adaptive growth with penalty loss **and patience**."""

        # inject constraints so `loss_fn` can access them --------------------
        if constraints is not None:
            self.eq = constraints["eq"]
            self.ineq = constraints["ineq"]

        device = next(self.parameters()).device
        patience = int(self.patience)
        inc = int(self.ex_k)
        dtmp = self.delta if delta is None else delta
        Δ = 0.0 if dtmp is None else float(dtmp)

        # reset failure counters at task start
        self._width_failures = 0
        self._depth_failures = 0

        best_snapshot = self.snapshot_state()
        best_val = float("inf")
        stop_outer = False

        # ─── OUTER: **WIDTH series** with patience ─────────────────────────
        while not stop_outer:
            # Baseline for width series (MSE-based) from last accepted best
            pre_width_snapshot = copy.deepcopy(self)
            pre_width_val = best_val
            self._width_failures = 0

            while True:
                # capacity guards for width
                total_neurons = sum(l.out_features for l in self.hidden_layers)
                curr_width = self.hidden_layers[-1].out_features
                if (
                    total_neurons + inc * len(self.hidden_layers) > self.max_neurons
                    or curr_width + inc > self.max_width
                ):
                    logger.info("Reached max_neurons / max_width — stopping width expansion.")
                    stop_outer = True
                    break

                # attempt a width increment
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
                    # ACCEPT width → reset counter & advance baseline
                    self._width_failures = 0
                    best_val = w_val
                    best_snapshot = self.snapshot_state()
                    pre_width_snapshot = copy.deepcopy(self)
                    pre_width_val = best_val
                    logger.info("Width expansion accepted; resetting failure counter.")

                    # ── After accepting width, run a **DEPTH series** with patience ──
                    pre_depth_snapshot = copy.deepcopy(self)
                    pre_depth_val = best_val
                    self._depth_failures = 0

                    while True:
                        if len(self.hidden_layers) >= self.max_depth:
                            logger.info("Reached max_depth — stopping depth expansion at this width.")
                            break

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
                            # ACCEPT depth → reset counter & advance baseline
                            self._depth_failures = 0
                            best_val = d_val
                            best_snapshot = self.snapshot_state()
                            pre_depth_snapshot = copy.deepcopy(self)
                            pre_depth_val = best_val
                            logger.info("Depth expansion accepted at current width; counter reset.")
                            # try another depth increment
                            continue
                        else:
                            # FAIL depth
                            self._depth_failures += 1
                            k, N = self._depth_failures, int(self.trials_depth)
                            if k < N:
                                logger.info(
                                    f"No val improvement after depth expansion (trial {k}/{N}); trying another depth expansion.")
                                continue  # keep added layer and try another depth step
                            else:
                                logger.info(
                                    f"Depth expansions without improvement reached {N}; rolling back depths and stopping depth series.")
                                self.restore_state(pre_depth_snapshot)
                                break  # exit depth series, return to width series

                    # after finishing (or skipping) depth series, continue width series
                    continue

                else:
                    # FAIL width
                    self._width_failures += 1
                    k, N = self._width_failures, int(self.trials_width)
                    if k < N:
                        logger.info(
                            f"No val improvement after width expansion (trial {k}/{N}); trying another width expansion.")
                        continue  # keep widened net and attempt another width increment
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back widths and stopping.")
                        self.restore_state(pre_width_snapshot)
                        stop_outer = True
                        break

        # —— final restore ——
        self.restore_state(best_snapshot)
        return best_val

    # ───────────────────── convenience ─────────────────────────────────────
    @property
    def lambdas(self) -> Tuple[float, float, float]:
        """Return *(λ_loss, λ_eq, λ_ineq)*."""
        return self.lambda_loss, self.lambda_equality, self.lambda_ineq


__all__ = [
    "PenaltyADPWidth",
    "_resize_linear",
    "_resize_head",
    "expand_width",
    "expand_depth",
]
