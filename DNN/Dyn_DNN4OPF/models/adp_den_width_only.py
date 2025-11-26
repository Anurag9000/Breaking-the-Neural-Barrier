"""
ADP-DEN (mask-free version) — width-only expansion with patience
────────────────────────────────────────────────────────────────
• Inner loop  : early-stopping on validation MSE (patience = self.patience)
• Outer loop  : widen ALL hidden layers by +ex_k neurons (keep depth) on plateau
                accept expansion iff best_val_mse < pre_exp_val_mse − delta
                allow up to `trials_depth` failed expansions before rollback
• Final state : restore snapshot with globally best (ΔP+ΔQ)

Notes
-----
- Mirrors `adp_den_depth_only.py` logic exactly; only the *expansion step* differs.
- Uses the same config keys (`delta`, `trials_depth`, `max_neurons`, `patience`).
- Adds `ex_k` (default: 16) to choose how many neurons to add per hidden layer.
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.models.dnn_den import DEN
from Dyn_DNN4OPF.models.dnn_den import (
    mean_constraint_violation,
    power_balance_residuals,
)
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ════════════════════════════════════════════════════════════════════════
# Resizers (mirror depth-only helpers; now used to *widen* layers)
# ════════════════════════════════════════════════════════════════════════

def _resize_linear(old: nn.Linear, new_out: int, new_in: int | None = None):
    """Return a new Linear(new_in, new_out) with overlapping weights copied."""
    if new_in is None:
        new_in = old.in_features
    new = nn.Linear(new_in, new_out, bias=True).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        new.bias[:r]       = old.bias[:r]
    return new


def _resize_head(old: nn.Linear, new_in: int):
    """Resize only the input dimension of the output head."""
    return _resize_linear(old, old.out_features, new_in)


def _cfg_get(cfg, key: str, default):
    """Dictionary-style get with attribute fallback."""
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


# ════════════════════════════════════════════════════════════════════════
# ADP-DEN (width-only growth)
# ════════════════════════════════════════════════════════════════════════
class ADP_DEN(DEN):
    # ─────────────────────────── init ───────────────────────────
    def __init__(self, config):
        super().__init__(config)
        # thresholds / patience knobs (keep names identical to depth-only code)
        self.delta = _cfg_get(config, "delta", 0.0)
        self.trials_depth = _cfg_get(config, "trials_depth", 10)  # outer patience (expansion failures)
        self._depth_failures = 0
        self.ex_k = int(_cfg_get(config, "ex_k", 16))             # neurons to add per hidden layer

        # optional output bounds (identical behavior to depth-only)
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(
                config.bounds_low, config.bounds_high, config.mask
            )
            self.bound_layer.apply_bounds.fill_(False)   # train/val
        else:
            self.bound_layer = nn.Identity()
            # keep interface parity with BoundedAct
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

    # ─────────────────────────── forward ───────────────────────────
    def forward(self, x):  # type: ignore[override]
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        x = self.head(x)
        return self.bound_layer(x)

    # ───────────────────────── training (plateau → widen with patience) ─────────────────────────
    def train_task(
        self, train_loader, val_loader, constraints, task_id, max_epochs=10_000, delta=None
    ) -> float:
        """
        Train with early stopping; when the val loss plateaus, try **width-only expansion**.
        Accept expansion iff best_val_mse < pre_exp_val_mse - delta (delta may be None → 0.0).
        Allow up to `trials_depth` consecutive failed expansions before rolling back to the
        pre-expansion baseline and stop expanding. Final snapshot chosen by (ΔP+ΔQ).

        Returns
        -------
        float
            Best validation MSE achieved for this task (at the final restored snapshot).
        """
        device         = next(self.parameters()).device
        orig_patience  = self.patience
        max_neurons    = int(_cfg_get(self.config, "max_neurons", 4096))
        delta          = self.delta if delta is None else (0.0 if delta is None else float(delta))
        eps            = 1e-12

        # keep task id for logging
        self.current_task = int(task_id)

        # fresh optimiser / scheduler for current architecture
        self.opt, self.scheduler = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        # global epoch counter across all expansions
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Phase-local trackers (best within the current architecture)
        best_snapshot = self.snapshot_state()
        best_val_mse  = float("inf")
        best_csum     = float("inf")  # ΔP+ΔQ at phase-best snapshot

        # Global best by constraints (for final restore)
        global_best_snap, global_best_csum = best_snapshot, float("inf")
        global_best_val_mse = float("inf")

        # Expansion acceptance baseline (last accepted state by MSE rule)
        exp_accept_snapshot = best_snapshot
        exp_accept_val_mse  = float("inf")

        # pre-exp baselines for the *current* expansion attempt/series
        pre_exp_snapshot = None
        pre_exp_val_mse  = None

        # remember starting widths for CSV logs
        fc1_start = self.layers[0].out_features
        fc2_start = self.layers[-1].out_features

        counter       = orig_patience
        stop_outer    = False
        just_expanded = False

        while not stop_outer:
            # ───────── inner loop (early stopping within current architecture) ─────────
            for _ in range(max_epochs):
                self.train()
                for xb, yb, *_ in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    self.opt.zero_grad(set_to_none=True)
                    loss = F.mse_loss(self(xb), yb)
                    loss.backward()
                    self.opt.step()

                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

                # evaluate on val
                self.eval()
                with torch.no_grad():
                    val_mse, physics = self._evaluate_loader(val_loader, constraints, device)
                csum = physics["ΔP"] + physics["ΔQ"]

                improved = val_mse < best_val_mse - eps
                if improved:
                    best_val_mse = val_mse
                    best_csum    = csum
                    best_snapshot = self.snapshot_state()
                    counter = orig_patience
                else:
                    counter -= 1

                # track global best by physics (for final restore)
                if csum < global_best_csum - eps:
                    global_best_csum   = csum
                    global_best_snap   = self.snapshot_state()
                    global_best_val_mse = min(global_best_val_mse, val_mse)

                self.global_epoch_count += 1
                logger.info(f"Epoch {self.global_epoch_count}: val-MSE = {val_mse:.6f} patience = {counter}")

                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break

                if counter <= 0:
                    # plateau reached in the current architecture
                    break

            if stop_outer:
                break

            # ───────── accept/retry/rollback for the *previous* expansion ─────────
            if just_expanded:
                if best_val_mse < pre_exp_val_mse - delta:
                    # success → reset counter and advance acceptance baseline
                    self._depth_failures = 0
                    exp_accept_snapshot = best_snapshot
                    exp_accept_val_mse  = best_val_mse
                    logger.info("Width expansion accepted; resetting failure counter.")
                    just_expanded = False
                else:
                    # failure → increment and decide whether to retry or rollback
                    self._depth_failures += 1
                    trial_k = self._depth_failures
                    N = int(self.trials_depth)
                    if trial_k < N:
                        logger.info(
                            f"No val improvement after width expansion (trial {trial_k}/{N}); trying another expansion.")
                        just_expanded = False  # we'll expand again below
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back and stopping.")
                        # rollback to pre-expansion baseline and stop expanding
                        self.restore_state(pre_exp_snapshot)
                        break

            # ───────── capacity guard before attempting another expansion ─────────
            if any(layer.out_features >= max_neurons for layer in self.layers):
                logger.info("Maximum capacity reached – stop expanding.")
                break

            # ───────── actually *widen* all hidden layers (width-only) ─────────
            # Record across-phase baseline (pre-expansion) for accept/reject
            pre_exp_snapshot = exp_accept_snapshot
            pre_exp_val_mse  = exp_accept_val_mse

            self._widen_all_hidden(step=self.ex_k, max_neurons=max_neurons, device=device)
            logger.info(f"Expanded width by +{self.ex_k} neurons on each hidden layer.")

            # new optimiser & scheduler so the fresh parameters get updated
            self.opt, self.scheduler = get_optimizer_scheduler(
                self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
            )

            # reset inner-loop bookkeeping for the enlarged model
            best_val_mse  = float("inf")
            best_csum     = float("inf")
            counter       = orig_patience
            just_expanded = True
            # loop continues → inner training on the bigger net

        # ensure we end with the best weights ever seen (by ΔP+ΔQ)
        self.restore_state(global_best_snap)
        # CSV log for the task
        self._csv_append({"val_loss": global_best_val_mse}, fc1_start, fc2_start, len(train_loader), len(val_loader))
        return global_best_val_mse

    # ───────────────────────── width grow helper ─────────────────────────
    def _widen_all_hidden(self, step: int, max_neurons: int, device):
        """Increase out_features of *every* hidden layer by `step` (capped by `max_neurons`).
        Propagate input dims forward and resize head to match the final hidden width.
        """
        prev_out = None
        for i, layer in enumerate(self.layers):
            new_in  = layer.in_features if prev_out is None else prev_out
            new_out = min(max_neurons, layer.out_features + step)
            self.layers[i] = _resize_linear(layer, new_out, new_in).to(device)
            prev_out = new_out
        # match head input to last hidden
        if self.head.in_features != prev_out:
            self.head = _resize_head(self.head, prev_out).to(device)

    # ───────────────────────── evaluation ─────────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()

        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device)), Yb.to(device)).item()
        val_loss = tot_mse / len(loader)

        # collect preds for constraint metrics
        X_all, Y_all = [], []
        with torch.no_grad():
            for Xb, *_ in loader:
                Xb = Xb.to(device)
                X_all.append(Xb)
                Y_all.append(self(Xb))
        X_all = torch.cat(X_all, dim=0)
        Y_all = torch.cat(Y_all, dim=0)

        # compute physics residuals + mean constraint violations
        with torch.no_grad():
            P_res, Q_res = power_balance_residuals(
                Y_all,
                y_bus=constraints["y_bus"],
                gen_bus_idx=constraints["gen_bus_idx"],
                load_bus_idx=constraints["load_bus_idx"],
                num_buses=self.n_bus,
            )
            ΔP, ΔQ, PG, QG, VM = mean_constraint_violation(
                P_res=P_res, Q_res=Q_res,
                y_bounds=constraints["ineq"],
                num_gens=self.n_gen, num_buses=self.n_bus,
            )
        return val_loss, {"ΔP": ΔP, "ΔQ": ΔQ, "PG": PG, "QG": QG, "VM": VM}

    # ───────────────────── snapshot helpers ─────────────────────
    def snapshot_state(self):
        return dict(
            state_dict   = self.state_dict(),
            hidden_sizes = [l.out_features for l in self.layers],
        )

    def restore_state(self, snap):
        dev          = next(self.parameters()).device
        target_sizes = snap["hidden_sizes"]

        # If depth differs, rebuild the stack to match snapshot (depth-friendly)
        cur_in = self.layers[0].in_features
        if len(target_sizes) != len(self.layers):
            new_layers = nn.ModuleList()
            prev_in = cur_in
            for tgt_out in target_sizes:
                new_layers.append(nn.Linear(prev_in, tgt_out, bias=True).to(dev))
                prev_in = tgt_out
            self.layers = new_layers
            if self.head.in_features != prev_in:
                self.head = _resize_head(self.head, prev_in)
        else:
            prev_out = None
            for i, (layer, tgt_out) in enumerate(zip(self.layers, target_sizes)):
                tgt_in = layer.in_features if prev_out is None else prev_out
                if layer.out_features != tgt_out or layer.in_features != tgt_in:
                    self.layers[i] = _resize_linear(layer, tgt_out, tgt_in).to(dev)
                prev_out = tgt_out
            if self.head.in_features != prev_out:
                self.head = _resize_head(self.head, prev_out)

        self.load_state_dict(snap["state_dict"], strict=True)

    # ────────────────────── _csv_append (log first & last) ─────────────────────
    def _csv_append(self, row, fc1_start, fc2_start, n_tr, n_val):
        """
        Keep CSV format identical, but map “fc1” → first hidden layer,
        “fc2” → last hidden layer (works for any depth ≥1).
        """
        log_dir = os.path.join(
            "Results",
            f"{self.config.model}_{self.config.case_name}",
            "logs",
        )
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir,
            f"log_task{self.current_task}_case-{self.config.case_name}.csv",
        )

        minimal = {
            "task":           self.current_task,
            "fc1_start":      fc1_start,
            "fc2_start":      fc2_start,
            "fc1_end":        self.layers[0].out_features,
            "fc2_end":        self.layers[-1].out_features,
            "val_loss":       row["val_loss"],
        }
        df = pd.DataFrame([minimal])
        df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
