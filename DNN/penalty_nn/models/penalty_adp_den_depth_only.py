"""
Penalty-ADP-DEN (depth-only expansion) + expansion patience
────────────────────────────────────────────────────────────
• Inner loop  : early-stopping on validation MSE
• Outer loop  : append one hidden layer (same width as current last layer)
                accept expansion iff best_val_mse < pre_exp_val_mse - delta
                allow up to `trials_depth` consecutive failed expansions before rollback
                final model = globally best snapshot by val MSE.

Loss (train only):
    L = λ_mse·MSE + λ_eq·(‖ΔP‖₁+‖ΔQ‖₁)_mean + λ_ineq·mean(bound_violation)
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
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
from Dyn_DNN4OPF.utils.constraint_losses import (
    power_balance_residuals,
    mean_constraint_violation,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ───────────────────────── helpers ─────────────────────────

def _cfg_get(cfg, key, default):
    """Dictionary-style get with attribute fallback."""
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _resize_linear(old: nn.Linear, new_out: int, new_in: int | None = None):
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
    return _resize_linear(old, old.out_features, new_in)


class PenaltyADP_DEN(DEN):
    """
    Depth-only expansion + penalty training with *patience on expansion*; single shared head.
    """

    # ─────────────────────────── init ───────────────────────────
    def __init__(
        self,
        config,
        *,
        lambda_mse: float = 1.0,
        lambda_eq: float = 1.0,
        lambda_ineq: float = 1.0,
    ):
        super().__init__(config)
        self.delta = _cfg_get(config, "delta", 0.0)
        self.trials_depth = _cfg_get(config, "trials_depth", 5)
        self._depth_failures = 0

        self.lambda_mse = float(lambda_mse)
        self.lambda_eq = float(lambda_eq)
        self.lambda_ineq = float(lambda_ineq)

        # optional output bounds layer (kept off during train/val)
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(
                config.bounds_low, config.bounds_high, config.mask
            )
            self.bound_layer.apply_bounds.fill_(False)
        else:
            self.bound_layer = nn.Identity()
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

    # ───────────────────────── forward ─────────────────────────
    def forward(self, x):
        y = x
        for layer in self.layers:
            y = F.relu(layer(y))
        y = self.head(y)
        return self.bound_layer(y)

    # ───────────────────────── penalty loss (train) ─────────────────────────
    def _penalty_loss(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """
        L = λ_mse·MSE + λ_eq·(‖ΔP‖₁+‖ΔQ‖₁)_mean + λ_ineq·mean(bound_violation)
        Equality terms from AC power balance; inequalities from box bounds.
        """
        y_pred = self.forward(x)

        # base MSE (per-batch mean)
        base_mse = F.mse_loss(y_pred, y_true)

        # slice outputs
        pg = y_pred[:, : self.n_gen]
        qg = y_pred[:, self.n_gen : 2 * self.n_gen]
        va = y_pred[:, 2 * self.n_gen : 2 * self.n_gen + self.n_bus]
        vm = y_pred[:, 2 * self.n_gen + self.n_bus : 2 * (self.n_gen + self.n_bus)]

        # inputs needed for power balance
        pd = x[:, : self.n_bus]
        qd = x[:, self.n_bus : 2 * self.n_bus]

        # equality residuals (AC power balance)
        resP, resQ = power_balance_residuals(
            pg, qg, pd, qd, vm, va,
            self.Ybus.to(x.device),
            self.bus_i_gen.to(x.device),
            self.gen_i_bus.to(x.device),
            self.gen_i_gen.to(x.device),
        )
        # average L1 magnitude across batch
        eq_term = resP.abs().mean() + resQ.abs().mean()

        # inequality violations (componentwise mean violation)
        pg_v = mean_constraint_violation(pg, self.pg_min, self.pg_max)
        qg_v = mean_constraint_violation(qg, self.qg_min, self.qg_max)
        vm_v = mean_constraint_violation(vm, self.vm_min, self.vm_max)
        ineq_term = (pg_v + qg_v + vm_v) / 3.0

        return (
            self.lambda_mse * base_mse
            + self.lambda_eq * eq_term
            + self.lambda_ineq * ineq_term
        )

    # ───────────────────────── training (plateau → expand with patience) ─────────────────────────
    def train_task(
        self, train_loader, val_loader, constraints, task_id, max_epochs=10_000, delta=None
    ) -> float:
        """
        Train with early stopping; on plateau, try **depth-only expansion**.
        Accept an expansion iff *validation MSE* improves vs the *pre-expansion baseline* by ≥ delta.
        Allow up to `trials_depth` consecutive failed expansions before rolling back to the baseline and stopping.
        Return the best validation MSE seen (globally accepted).
        """
        device         = next(self.parameters()).device
        orig_patience  = self.patience
        # resolve delta (None → 0.0)
        dtmp           = self.delta if delta is None else delta
        delta_thr      = 0.0 if dtmp is None else float(dtmp)
        max_neurons    = self.config.max_neurons
        eps            = 1e-12

        self.current_task = task_id
        # reset failure counter at the start of the task
        self._depth_failures = 0

        # fresh optimiser / scheduler for the current architecture
        self.opt, self.scheduler = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # per-phase trackers
        best_snapshot = self.snapshot_state()
        best_val_mse  = float("inf")

        # global accepted best by val MSE
        accepted_snapshot = best_snapshot
        accepted_val_mse  = float("inf")

        counter       = orig_patience
        stop_outer    = False
        just_expanded = False

        # remember starting widths for logs
        fc1_start = self.layers[0].out_features
        fc2_start = self.layers[-1].out_features

        # pre-expansion baseline for the *current* series (updated only on ACCEPT)
        pre_exp_snapshot = accepted_snapshot
        pre_exp_val_mse  = accepted_val_mse

        while not stop_outer:
            # ───────── inner loop (early stopping) ─────────
            for _ in range(max_epochs):
                self.train()
                for xb, yb, _ in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    self.opt.zero_grad(set_to_none=True)
                    loss = self._penalty_loss(xb, yb)
                    loss.backward()
                    self.opt.step()

                if self.scheduler is not None:
                    try:
                        self.scheduler.step()
                    except Exception:
                        pass

                # validation (pure MSE)
                self.eval()
                val_sum = 0.0
                with torch.no_grad():
                    for xb, yb, _ in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        val_sum += F.mse_loss(self(xb), yb).item()
                val_mse = val_sum / len(val_loader)

                logger.info(f"Epoch {self.global_epoch_count}: val-MSE = {val_mse:.6f} patience = {counter}")

                improved = val_mse < best_val_mse - eps
                if improved:
                    best_val_mse  = val_mse
                    best_snapshot = self.snapshot_state()
                    counter       = orig_patience
                else:
                    counter -= 1

                # track global accepted best
                if val_mse < accepted_val_mse - eps:
                    accepted_val_mse  = val_mse
                    accepted_snapshot = self.snapshot_state()

                self.global_epoch_count += 1

                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break
                if counter == 0:
                    break  # plateau → consider expansion

            # ensure we’re sitting at the phase-best weights
            self.restore_state(best_snapshot)

            if stop_outer:
                break

            # ───────── post-plateau decision for the *last* expansion attempt ─────────
            if just_expanded:
                # accept only if improved vs pre-expansion baseline by ≥ delta
                if best_val_mse < pre_exp_val_mse - delta_thr:
                    # ACCEPT: keep enlarged net, reset failure counter, and advance the baseline
                    self._depth_failures = 0
                    logger.info("Depth expansion accepted; resetting failure counter.")
                    pre_exp_snapshot = accepted_snapshot
                    pre_exp_val_mse  = accepted_val_mse
                    just_expanded    = False
                else:
                    # FAIL: increment and decide retry vs rollback
                    self._depth_failures += 1
                    k = self._depth_failures
                    N = int(self.trials_depth)
                    if k < N:
                        logger.info(
                            f"No val improvement after expansion (trial {k}/{N}); trying another expansion.")
                        # keep architecture; we'll attempt another depth expansion immediately
                        just_expanded = False
                    else:
                        logger.info(
                            f"Expansions without improvement reached {N}; rolling back and stopping.")
                        # patience exhausted → rollback to the pre-expansion baseline and stop
                        self.restore_state(pre_exp_snapshot)
                        break

            # capacity guard: if any hidden layer already at max width, stop expanding
            if any(layer.out_features >= max_neurons for layer in self.layers):
                logger.info("Maximum capacity reached – stop expanding.")
                break

            # ───────── actually append a new hidden layer (depth-only) ─────────
            # NOTE: do *not* overwrite pre_exp_* here on failures; it's advanced only upon ACCEPT above
            new_width  = self.layers[-1].out_features  # minimal-change: reuse last width
            in_width   = new_width
            self.layers.append(nn.Linear(in_width, new_width, bias=True).to(device))
            self.head = _resize_head(self.head, new_width)
            logger.info(f"Expanded depth by +1 layer (width={new_width}).")

            # new optimiser / scheduler for the new params
            self.opt, self.scheduler = get_optimizer_scheduler(
                self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
            )

            # reset inner-loop bookkeeping for the enlarged model
            best_val_mse  = float("inf")
            counter       = orig_patience
            just_expanded = True
            # continue to next inner loop

        # end outer while — ensure we end with globally accepted best
        self.restore_state(accepted_snapshot)
        return accepted_val_mse

    # ───────────────────────── evaluation helper ─────────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()
        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device)), Yb.to(device)).item()
        val_loss = tot_mse / len(loader)

        # (unchanged) physics metrics if you need them for debugging
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
        vm = Y_val[:, 2 * self.n_gen + self.n_bus : 2 * (self.n_gen + self.n_bus)]
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
            state_dict   = self.state_dict(),
            hidden_sizes = [l.out_features for l in self.layers],
        )

    def restore_state(self, snap):
        dev          = next(self.parameters()).device
        target_sizes = snap["hidden_sizes"]

        # rebuild stack to match snapshot depth (depth-friendly)
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

    # ────────────────────── CSV logging (first & last) ─────────────────────
    def _csv_append(self, row, fc1_start, fc2_start, n_tr, n_val):
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


__all__ = ["PenaltyADP_DEN", "_resize_linear", "_resize_head"]
