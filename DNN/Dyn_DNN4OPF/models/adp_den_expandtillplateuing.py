"""
ADP-DEN  (mask-free version) + expansion patience (width-only)
────────────────────────────
• Inner loop  : early-stopping on validation MSE
• Outer loop  : add `ex_k` neurons to all hidden layers (width expansion)
                accept expansion iff best_val_mse < pre_exp_val_mse - delta
                allow up to `trials_width` consecutive failed expansions before rollback
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
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


# ════════════════════════════════════════════════════════════════════════
# ADP-DEN  (no masks / timestamps) with width-patience
# ════════════════════════════════════════════════════════════════════════
class ADP_DEN(DEN):
    # ─────────────────────────── init ───────────────────────────
    def __init__(self, config):
        super().__init__(config)
        # thresholds / patience knobs
        self.delta = _cfg_get(config, "delta", 0.0)
        self.trials_width = _cfg_get(config, "trials_width", 5)
        # failure counter (width only in this model)
        self._width_failures = 0

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
        for layer in self.layers:          # loop through all hidden layers
            x = F.relu(layer(x))
        x = self.head(x)
        return self.bound_layer(x)

    # ───────────────────────── training with adaptive width expansion + patience ──────────────────────────
    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader,
        constraints,
        *,
        max_epochs: int = 10_000,
        delta: float | None = None,
    ) -> float:
        """
        Inner loop  : vanilla early stopping on validation MSE.
        Outer loop  : when the inner loop plateaus, widen **every** hidden layer by
                      ``ex_k`` neurons and restart training.
        Expansion is accepted IFF best_val_mse < pre_exp_val_mse - delta, where
        delta is taken from config (None → 0.0). Allow up to `trials_width` consecutive
        failed expansions before rolling back to the pre-expansion baseline and stopping.

        The final restored snapshot is the globally best by (ΔP+ΔQ), unchanged from original.
        Returns
        -------
        float
            Best validation MSE achieved at the final (restored) snapshot.
        """
        device         = next(self.parameters()).device
        orig_patience  = self.patience
        ex_k           = self.ex_k
        delta          = self.delta if delta is None else (0.0 if delta is None else float(delta))
        max_neurons    = _cfg_get(self.config, "max_neurons", getattr(self, "max_neurons", 10_000))
        eps            = 1e-12  # strict improvement tolerance for book-keeping

        # reset failure counter at the start of the task
        self._width_failures = 0

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

        # Global across-phase best by constraints (final target)
        # Seed using current unexpanded model
        _cur_val_loss, _cur_cmetrics = self._evaluate_loader(val_loader, constraints, device)
        accepted_snapshot = best_snapshot
        accepted_csum     = float(_cur_cmetrics["ΔP"] + _cur_cmetrics["ΔQ"])
        accepted_val_mse  = _cur_val_loss

        # Baseline for expansion acceptance (MSE-based)
        exp_accept_snapshot = accepted_snapshot
        exp_accept_val_mse  = accepted_val_mse

        # Pre-expansion baseline for the *current* expansion series (MSE-based)
        pre_exp_snapshot  = None
        pre_exp_val_mse   = None

        just_expanded     = False
        stop_outer        = False

        # ───────────────────────── outer loop ─────────────────────────
        while not stop_outer:

            # -------- inner early-stopping loop --------
            counter = orig_patience
            while counter > 0:
                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break

                # ---- training step ----
                self.train()
                for xb, yb, _ in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    loss = F.mse_loss(self(xb), yb)
                    self.opt.zero_grad(set_to_none=True)
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

                # Track phase-best by MSE (early-stopping uses eps, not delta)
                if val_mse < best_val_mse - eps:
                    best_val_mse  = val_mse
                    best_snapshot = self.snapshot_state()
                    # compute constraints (ΔP+ΔQ) at this new phase-best snapshot
                    self.restore_state(best_snapshot)
                    _, _cm = self._evaluate_loader(val_loader, constraints, device)
                    best_csum = float(_cm["ΔP"] + _cm["ΔQ"])
                    counter   = orig_patience
                else:
                    counter -= 1

                # update globally accepted best by constraints as we go
                if best_csum < accepted_csum - eps:
                    accepted_csum     = best_csum
                    accepted_snapshot = best_snapshot
                    accepted_val_mse  = best_val_mse

            # reached max_epochs during inner loop?
            if stop_outer:
                break

            # -------- plateau reached → accept/reject (if coming from an expansion), then maybe expand --------
            self.restore_state(best_snapshot)      # roll back to phase-best weights

            if just_expanded:
                # Decide using MSE vs pre-expansion baseline
                accept = best_val_mse < (pre_exp_val_mse - (0.0 if delta is None else float(delta)))
                if accept:
                    # success → reset counter and advance acceptance baseline
                    self._width_failures = 0
                    exp_accept_snapshot = best_snapshot
                    exp_accept_val_mse  = best_val_mse
                    logger.info("Width expansion accepted; resetting failure counter.")
                    just_expanded = False
                else:
                    # failure → increment and decide whether to retry or rollback
                    self._width_failures += 1
                    trial_k = self._width_failures
                    N = int(self.trials_width)
                    if trial_k < N:
                        logger.info(
                            f"No val improvement after width expansion (trial {trial_k}/{N}); trying another expansion.")
                        just_expanded = False  # we'll expand again below (keep architecture, same baseline)
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back and stopping.")
                        # rollback to pre-expansion baseline and stop expanding
                        self.restore_state(pre_exp_snapshot)
                        # also keep acceptance baseline consistent
                        exp_accept_snapshot = pre_exp_snapshot
                        exp_accept_val_mse  = pre_exp_val_mse
                        break  # exit outer while → end training

            # ───────── capacity guard before attempting another expansion ─────────
            # stop if ANY hidden layer is already at or would exceed the limit after widening
            if any(layer.out_features >= max_neurons for layer in self.layers):
                logger.info("Maximum capacity reached – stop expanding.")
                break

            # ---- actually enlarge each hidden layer ----
            # Record pre-expansion baseline (MSE-based) for accept/reject for *this* series
            pre_exp_snapshot = exp_accept_snapshot
            pre_exp_val_mse  = exp_accept_val_mse

            prev_out = None
            for i, layer in enumerate(self.layers):
                new_out = layer.out_features + ex_k
                new_in  = layer.in_features if prev_out is None else prev_out
                # guard: if widening would exceed max_neurons, abort expansion cleanly
                if new_out > max_neurons:
                    logger.info("Proposed width exceeds max_neurons – stop expanding.")
                    stop_outer = True
                    break
                self.layers[i] = _resize_linear(layer, new_out, new_in)
                prev_out = new_out
            if stop_outer:
                break

            self.head = _resize_head(self.head, prev_out)
            logger.info(f"Expanded all {len(self.layers)} hidden layers by {ex_k} neurons.")

            # new optimiser & scheduler so the fresh parameters get updated
            self.opt, self.scheduler = get_optimizer_scheduler(
                self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
            )

            # reset inner-loop bookkeeping for the enlarged model
            best_val_mse  = float("inf")
            best_csum     = float("inf")
            counter       = orig_patience
            just_expanded = True
            # loop continues → inner training on the wider net

        # ensure we end with the best weights ever seen (by ΔP+ΔQ)
        self.restore_state(accepted_snapshot)
        return accepted_val_mse

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
            y_bus=constraints["eq"]["y_bus"],
            gen_bus_idx=constraints["eq"]["gen_bus_idx"],
            load_bus_idx=constraints["eq"]["load_bus_idx"],
            n_bus=self.n_bus,
        )

        ΔP, ΔQ, PG, QG, VM = mean_constraint_violation(
            Y_pred=Y_val,
            res_real=resP,
            res_imag=resQ,
            bounds=constraints["ineq"],
            num_gens=self.n_gen,
            num_buses=self.n_bus,
        )

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
        assert len(target_sizes) == len(self.layers), "Depth mismatch with snapshot."

        prev_out = None
        for i, (layer, tgt_out) in enumerate(zip(self.layers, target_sizes)):
            tgt_in = layer.in_features if prev_out is None else prev_out
            if layer.out_features != tgt_out or layer.in_features != tgt_in:
                self.layers[i] = _resize_linear(layer, tgt_out, tgt_in).to(dev)
            prev_out = tgt_out

        if self.head.in_features != prev_out:           # keep head in sync
            self.head = _resize_head(self.head, prev_out)
        self.load_state_dict(snap["state_dict"], strict=True)

    def _save_snapshot(self):
        self._snapshot = self.snapshot_state()

    def _load_snapshot(self):
        self.restore_state(self._snapshot)

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
