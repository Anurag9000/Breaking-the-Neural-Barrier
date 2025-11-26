"""
penalty_adp_den.py + expansion patience (width series)
───────────────────────────────────────────────────────
Penalty‑augmented replica of *ADP‑DEN (expand‑till‑plateau)* that keeps **every
single feature** (dynamic expansion, optional bounded‑activation clipping,
early‑stopping, etc.) while replacing the plain MSE loss with a weighted
penalty loss and adding **patience on expansion**:

  • Accept an expansion IFF best_val_mse < pre_exp_val_mse − delta
  • Allow up to `trials_width` consecutive failed expansions before rollback
  • On failure within patience: keep the added capacity and immediately try
    another expansion of the same dimension (width)

The class remains a **drop‑in replacement** for `ADP_DEN` (same public API).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

# ── internal imports (unchanged from original) ─────────────────────────────
from Dyn_DNN4OPF.models.adp_den_expandtillplateuing import (
    ADP_DEN as _BaseADP_DEN,
    _resize_linear,  # re‑exported for callers that resize manually
    _resize_head,
)
from Dyn_DNN4OPF.models.dnn_den import (
    power_balance_residuals,
    mean_constraint_violation,
)
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS

import logging
logger = logging.getLogger(__name__)

# ───────────────────────── helpers ─────────────────────────

def _cfg_get(cfg, key, default):
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)

# ════════════════════════════════════════════════════════════════════════
#  Main class
# ════════════════════════════════════════════════════════════════════════
class PenaltyADP_DEN(_BaseADP_DEN):
    """ADP‑DEN with penalty‑augmented loss and **patience on width expansion**."""

    # ─────────────────── Init ───────────────────
    def __init__(
        self,
        config,
        *,
        lambda_loss: float = 1.0,
        lambda_equality: float = 1.0,
        lambda_ineq: float = 1.0,
        **unused,
    ) -> None:
        super().__init__(config)

        # expose penalty weights *inside the model* (independent control)
        self.lambda_loss = lambda_loss
        self.lambda_equality = lambda_equality
        self.lambda_ineq = lambda_ineq

        # store constraint dicts directly on self so `loss_fn` needs no kwargs
        self.eq = getattr(config, "eq", {})
        self.ineq = getattr(config, "ineq", {})

        # patience knob for WIDTH expansions (dict-like access with default)
        self.trials_width = _cfg_get(config, "trials_width", 5)
        self._width_failures = 0

        # delta/threshold (may be None → treat as 0.0 at use-sites)
        self.delta = _cfg_get(config, "delta", 0.0)

    # ─────────────────── Loss ───────────────────
    def loss_fn(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Penalty loss: λ₁·MSE + λ₂·‖ΔP,ΔQ‖₂ + λ₃·‖g⁺‖₂."""
        y_pred = self(x)
        base_mse = F.mse_loss(y_pred, y_true)

        # --- convenience slices -------------------------------------------------
        n_bus = x.size(1) // 2
        n_gen = self.n_gen
        pd, qd = x[:, :n_bus], x[:, n_bus : 2 * n_bus]
        pg = y_pred[:, :n_gen]
        qg = y_pred[:, n_gen : 2 * n_gen]
        va = y_pred[:, 2 * n_gen : 2 * n_gen + n_bus]
        vm = y_pred[:, 2 * n_gen + n_bus : 2 * (n_gen + n_bus)]

        # --- equality residuals (per‑bus) --------------------------------------
        res_P, res_Q = power_balance_residuals(
            pg=pg,
            qg=qg,
            pd=pd,
            qd=qd,
            vm=vm,
            va=va,
            y_bus=self.eq["y_bus"],
            gen_bus_idx=self.eq["gen_bus_idx"],
            load_bus_idx=self.eq["load_bus_idx"],
            n_bus=n_bus,
        )
        eq_viols = torch.norm(res_P, dim=1) + torch.norm(res_Q, dim=1)  # [B]

        # --- inequality residuals (bounds) --------------------------------------
        *_unused, ineq_viols = mean_constraint_violation(
            Y_pred=y_pred,
            res_real=res_P,
            res_imag=res_Q,
            bounds=self.ineq,
            num_gens=n_gen,
            num_buses=n_bus,
        )  # returns scalar mean viol per‑sample

        return (
            self.lambda_loss * base_mse
            + self.lambda_equality * eq_viols.mean()
            + self.lambda_ineq * ineq_viols.mean()
        )

    # ─────────────────── Training (override) ───────────────────
    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader,
        constraints: dict,
        *,
        max_epochs: int = 10_000,
        delta: float | None = None,
    ) -> float:
        """Same algorithm as *ADP‑DEN.fit_task* but back‑propagates the penalty loss
        and uses **patience on width expansions** with a delta acceptance threshold.
        """

        # expose constraints to `loss_fn`
        self.eq = constraints["eq"]
        self.ineq = constraints["ineq"]

        import math
        device = next(self.parameters()).device
        orig_patience = self.patience
        ex_k = self.ex_k
        # resolve delta (None → 0.0) for acceptance comparisons
        dtmp = self.delta if delta is None else delta
        delta_thr = 0.0 if dtmp is None else float(dtmp)

        # reset failure counter at task start
        self._width_failures = 0

        # fresh optimiser/scheduler every (re)start -----------------------------
        self.opt, self.scheduler = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Phase-local trackers
        best_snapshot = self.snapshot_state()
        best_val_mse = math.inf

        # Global across-phase best (by val MSE)
        accepted_snapshot = best_snapshot
        accepted_val_mse = math.inf

        # Pre-expansion baseline for the current width series (MSE-based)
        pre_exp_snapshot = accepted_snapshot
        pre_exp_val_mse = accepted_val_mse
        in_post_expansion = False

        stop_outer = False

        # ─── OUTER expansion loop ──────────────────────────────────────────────
        while not stop_outer:
            counter = orig_patience
            # —— INNER early‑stopping loop ——
            while counter > 0:
                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break

                # — training epoch —
                self.train()
                for xb, yb, _ in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    loss = self.loss_fn(xb, yb)  # ← penalty loss
                    self.opt.zero_grad()
                    loss.backward()
                    self.opt.step()
                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

                # — validation epoch (pure MSE) —
                self.eval()
                val_sum = 0.0
                with torch.no_grad():
                    for xb, yb, _ in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        val_sum += F.mse_loss(self(xb), yb).item()
                val_mse = val_sum / max(1, len(val_loader))

                if val_mse < best_val_mse - 1e-12:  # strict improvement for ES
                    best_val_mse = val_mse
                    best_snapshot = self.snapshot_state()
                    counter = orig_patience
                else:
                    counter -= 1

                # track global accepted best
                if best_val_mse < accepted_val_mse - 1e-12:
                    accepted_val_mse = best_val_mse
                    accepted_snapshot = best_snapshot

            if stop_outer:
                break

            # —— plateau → accept/reject last expansion (if any), then maybe expand ——
            self.restore_state(best_snapshot)

            if in_post_expansion:
                # Decide acceptance based on improvement vs pre-expansion baseline
                if best_val_mse < pre_exp_val_mse - delta_thr:
                    # ACCEPT: reset failure counter & advance baseline
                    self._width_failures = 0
                    logger.info("Width expansion accepted; resetting failure counter.")
                    pre_exp_snapshot = accepted_snapshot
                    pre_exp_val_mse = accepted_val_mse
                    in_post_expansion = False
                else:
                    # FAIL: increment and decide retry vs rollback
                    self._width_failures += 1
                    k, N = self._width_failures, int(self.trials_width)
                    if k < N:
                        logger.info(
                            f"No val improvement after width expansion (trial {k}/{N}); trying another expansion.")
                        in_post_expansion = False  # keep architecture and try another widen
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back and stopping.")
                        # rollback to the pre-expansion baseline and stop expanding
                        self.restore_state(pre_exp_snapshot)
                        break

            # capacity guard for two‑layer topology (stop if either at max_neurons)
            if self.fc1.out_features >= self.config.max_neurons or self.fc2.out_features >= self.config.max_neurons:
                break

            # record baseline (pre-expansion) for accept/reject — use LAST ACCEPTED
            pre_exp_snapshot = accepted_snapshot
            pre_exp_val_mse = accepted_val_mse

            # —— actually expand both hidden layers (width) ——
            self.fc1 = _resize_linear(self.fc1, self.fc1.out_features + self.ex_k)
            self.layers[0] = self.fc1
            self.fc2 = _resize_linear(
                self.fc2,
                self.fc2.out_features + self.ex_k,
                self.fc1.out_features,
            )
            self.layers[1] = self.fc2
            self.head = _resize_head(self.head, self.fc2.out_features)
            logger.info(
                f"Expanded width by +{self.ex_k} on both hidden layers (fc1={self.fc1.out_features}, fc2={self.fc2.out_features})."
            )

            # new optimiser for new params
            self.opt, self.scheduler = get_optimizer_scheduler(
                self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
            )

            # reset inner‑loop state for the enlarged net
            best_val_mse = float("inf")
            counter = orig_patience
            in_post_expansion = True
            # continue into next inner loop iteration with bigger net

        # end outer while — ensure we end with globally accepted best
        self.restore_state(accepted_snapshot)
        return accepted_val_mse

# Convenience re‑export so calling code can:  from penalty_adp_ditto import PenaltyADP_DEN
__all__ = ["PenaltyADP_DEN", "_resize_linear", "_resize_head"]
