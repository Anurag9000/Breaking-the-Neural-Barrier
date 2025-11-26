"""
ADP-DEN  (mask-free version) + expansion patience (depth-only)
────────────────────────────
• Inner loop  : early-stopping on validation MSE
• Outer loop  : append one hidden layer (depth-only)
                accept expansion iff best_val_mse < pre_exp_val_mse - delta
                allow up to `trials_depth` consecutive failed expansions before rollback
                final model = globally best (ΔP+ΔQ) snapshot (weights + architecture).

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
# basic linear-resize helpers
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
        return cfg.get(key, default)  # dict / EasyDict path
    except Exception:
        return getattr(cfg, key, default)


class ADP_DEN(DEN):
    # ─────────────────────────── init ───────────────────────────
    def __init__(self, config):
        super().__init__(config)
        # thresholds / patience knobs
        self.delta = _cfg_get(config, "delta", 0.0)
        self.trials_depth = _cfg_get(config, "trials_depth", 5)
        # failure counter (depth only in this model)
        self._depth_failures = 0

        # optional output bounds
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(
                config.bounds_low, config.bounds_high, config.mask
            )
            self.bound_layer.apply_bounds.fill_(False)   # train/val
        else:
            self.bound_layer = nn.Identity()
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

    # ───────────────────────── forward ─────────────────────────
    def forward(self, x):
        y = x
        for layer in self.layers:
            y = F.relu(layer(y))
        y = self.head(y)
        # always keep a toggle for test-time bounds
        return self.bound_layer(y)

    # ───────────────────────── training (plateau → expand with patience) ─────────────────────────
    def train_task(
        self, train_loader, val_loader, constraints, task_id, max_epochs=10_000, delta=None
    ) -> float:
        """
        Train with early stopping; when the val loss plateaus, try **depth-only expansion**.
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
        delta          = self.delta if delta is None else (0.0 if delta is None else float(delta))
        max_neurons    = _cfg_get(self.config, "max_neurons", getattr(self, "max_neurons", 10_000))
        eps            = 1e-12

        self.current_task = task_id
        # reset failure counter at the start of the task
        self._depth_failures = 0

        # fresh optimiser / scheduler for the current architecture
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

        counter       = orig_patience
        stop_outer    = False
        just_expanded = False
        # pre-exp baselines for the *current* expansion attempt/series
        pre_exp_snapshot = None
        pre_exp_val_mse  = None

        # remember starting widths for CSV logs
        fc1_start = self.layers[0].out_features
        fc2_start = self.layers[-1].out_features

        while not stop_outer:
            # ───────── inner loop (early stopping within current architecture) ─────────
            for _ in range(max_epochs):
                self.train()
                for xb, yb, _ in train_loader:
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

                # ---- validation step ----
                self.eval()
                val_sum = 0.0
                with torch.no_grad():
                    for xb, yb, _ in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        val_sum += F.mse_loss(self(xb), yb).item()
                val_mse = val_sum / len(val_loader)

                logger.info(f"Epoch {self.global_epoch_count}: val-MSE = {val_mse:.6f} patience = {counter}")

                # physics metrics on full val-set (for global best only)
                _cur_val_loss, physics = self._evaluate_loader(val_loader, constraints, device)
                csum = physics["ΔP"] + physics["ΔQ"]

                improved = val_mse < best_val_mse - eps
                if improved:
                    best_val_mse = val_mse
                    best_csum    = csum
                    best_snapshot = self.snapshot_state()
                    counter = orig_patience
                else:
                    counter -= 1

                self.global_epoch_count += 1

                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break
                if counter == 0:
                    break  # plateau in this architecture → move to accept/expand logic

            # ───────── update global best by constraints (final selection) ─────────
            _cur_val_loss, physics = self._evaluate_loader(val_loader, constraints, device)
            _cur_csum = physics["ΔP"] + physics["ΔQ"]
            if _cur_csum < global_best_csum - eps:
                global_best_snap   = best_snapshot
                global_best_csum   = _cur_csum
                global_best_val_mse = best_val_mse

            if stop_outer:
                break

            # ───────── post-plateau decision: accept / retry / rollback ─────────
            if just_expanded:
                # We just trained after an expansion → decide using MSE delta vs pre-exp baseline
                accept = best_val_mse < (pre_exp_val_mse - (0.0 if delta is None else float(delta)))
                if accept:
                    # success → reset counter and advance acceptance baseline
                    self._depth_failures = 0
                    exp_accept_snapshot = best_snapshot
                    exp_accept_val_mse  = best_val_mse
                    logger.info("Depth expansion accepted; resetting failure counter.")
                    just_expanded = False
                else:
                    # failure → increment and decide whether to retry or rollback
                    self._depth_failures += 1
                    trial_k = self._depth_failures
                    N = int(self.trials_depth)
                    if trial_k < N:
                        logger.info(
                            f"No val improvement after depth expansion (trial {trial_k}/{N}); trying another expansion.")
                        just_expanded = False  # we'll expand again below
                    else:
                        logger.info(
                            f"Depth expansions without improvement reached {N}; rolling back and stopping.")
                        # rollback to pre-expansion baseline and stop expanding
                        self.restore_state(pre_exp_snapshot)
                        # also keep acceptance baseline consistent
                        exp_accept_snapshot = pre_exp_snapshot
                        exp_accept_val_mse  = pre_exp_val_mse
                        break  # exit outer while → end training
            else:
                # First plateau with this architecture and no expansion yet →
                # establish the acceptance baseline from the current phase-best
                exp_accept_snapshot = best_snapshot
                exp_accept_val_mse  = best_val_mse

            # ───────── capacity guard before attempting another expansion ─────────
            if any(layer.out_features >= max_neurons for layer in self.layers):
                logger.info("Maximum capacity reached – stop expanding.")
                break

            # ───────── actually append a new hidden layer (depth-only) ─────────
            # Record across-phase baseline (pre-expansion) for accept/reject
            pre_exp_snapshot = exp_accept_snapshot
            pre_exp_val_mse  = exp_accept_val_mse

            # New layer width = current last hidden width (minimal-change policy)
            new_width  = self.layers[-1].out_features
            in_width   = new_width
            self.layers.append(nn.Linear(in_width, new_width, bias=True).to(device))
            self.head = _resize_head(self.head, new_width)
            logger.info(f"Expanded depth by +1 layer (width={new_width}).")

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
        return global_best_val_mse

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
        X_val, Y_val = torch.cat(X_all), torch.cat(Y_all)

        pg = Y_val[:, : self.n_gen]
        qg = Y_val[:, self.n_gen : 2 * self.n_gen]
        va = Y_val[:, 2 * self.n_gen : 2 * self.n_gen + self.n_bus]
        vm = Y_val[
            :, 2 * self.n_gen + self.n_bus : 2 * (self.n_gen + self.n_bus)
        ]
        pd = X_val[:, : self.n_bus]
        qd = X_val[:, self.n_bus : 2 * self.n_bus]

        resP, resQ = power_balance_residuals(
            pg, qg, pd, qd, vm, va,
            self.Ybus.to(device),
            self.bus_i_gen.to(device),
            self.gen_i_bus.to(device),
            self.gen_i_gen.to(device),
        )
        ΔP, ΔQ = resP.abs().mean().item(), resQ.abs().mean().item()
        PG     = mean_constraint_violation(pg, self.pg_min, self.pg_max).item()
        QG     = mean_constraint_violation(qg, self.qg_min, self.qg_max).item()
        VM     = mean_constraint_violation(vm, self.vm_min, self.vm_max).item()

        return val_loss, dict(ΔP=ΔP, ΔQ=ΔQ, PG=PG, QG=QG, VM=VM)

    # ───────────────────── snapshot helpers ─────────────────────
    def snapshot_state(self):
        return dict(
            state_dict   = self.state_dict(),                  # full weights/buffers
            hidden_sizes = [l.out_features for l in self.layers]  # width of every layer
        )

    def restore_state(self, snap):
        dev          = next(self.parameters()).device
        target_sizes = snap["hidden_sizes"]

        # If depth differs, rebuild the stack to match snapshot (depth-only friendly)
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
            if self.head.in_features != prev_out:           # keep head in sync
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
            "fc1_start":      fc1_start,                     # caller’s value
            "fc2_start":      fc2_start,                     # caller’s value
            "fc1_end":        self.layers[0].out_features,   # first hidden layer
            "fc2_end":        self.layers[-1].out_features,  # last  hidden layer
            "val_loss":       row["val_loss"],
        }
        df = pd.DataFrame([minimal])
        df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
