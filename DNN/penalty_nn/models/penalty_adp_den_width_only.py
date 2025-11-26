"""
Penalty-ADP-DEN (width-only expansion) + expansion patience
────────────────────────────────────────────────────────────
• Inner loop  : early-stopping on validation MSE
• Outer loop  : **widen all hidden layers** by +ex_k neurons (keep depth)
                accept expansion iff best_val_mse < pre_exp_val_mse − delta
                allow up to `trials_depth` consecutive failed expansions before rollback
                final model = globally accepted snapshot by validation MSE.

TRAINING loss (validation still MSE):
    L = λ_mse·MSE + λ_eq·(‖ΔP‖₁+‖ΔQ‖₁)_mean + λ_ineq·mean(bound_violation)

This mirrors `penalty_adp_den_depth_only.py` but replaces the *depth append* with a *width increase*.
"""

import os
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals, mean_constraint_violation
from Dyn_DNN4OPF.models.dnn_den import DEN
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


# ───────────────────────── resize helpers (width growth) ─────────────────────────
def _resize_linear(old: nn.Linear, new_out: int, new_in: int | None = None):
    if new_in is None:
        new_in = old.in_features
    new = nn.Linear(new_in, new_out, bias=True).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        new.bias[:r] = old.bias[:r]
    return new


def _resize_head(old: nn.Linear, new_in: int):
    return _resize_linear(old, old.out_features, new_in)


def _cfg_get(cfg, key: str, default):
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


class PenaltyADP_DEN(DEN):
    """Width-only expansion + penalty training with *patience on expansion*; single shared head."""

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
        self.trials_depth = _cfg_get(config, "trials_depth", 10)
        self._depth_failures = 0
        self.ex_k = int(_cfg_get(config, "ex_k", 16))  # neurons to add per hidden layer

        self.lambda_mse = float(lambda_mse)
        self.lambda_eq = float(lambda_eq)
        self.lambda_ineq = float(lambda_ineq)

        # optional bounds at test-time (train/val disabled)
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(config.bounds_low, config.bounds_high, config.mask)
            self.bound_layer.apply_bounds.fill_(False)
        else:
            self.bound_layer = nn.Identity()
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

    # ───────────────────────── forward ─────────────────────────
    def forward(self, x):
        y = x
        for i, layer in enumerate(self.layers):
            y = layer(y)
            if i < len(self.layers) - 1:
                y = F.relu(y)
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

        # equality residuals via power balance
        with torch.no_grad():
            x_full = torch.cat([pd, qd, pg, qg, va, vm], dim=-1)
        resP, resQ = power_balance_residuals(
            y_pred,  # predicted outputs suffice if your util reconstructs internally; else use x_full
            {
                "y_bus": self.y_bus if hasattr(self, "y_bus") else None,
                "gen_bus_idx": self.gen_bus_idx if hasattr(self, "gen_bus_idx") else None,
                "load_bus_idx": self.load_bus_idx if hasattr(self, "load_bus_idx") else None,
                "ineq": getattr(self, "ineq_bounds", None),
            },
        )
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

    # ───────────────────────── training (plateau → *widen*) ─────────────────────────
    def train_task(self, train_loader, val_loader, constraints, task_id, max_epochs=10_000, delta=None) -> float:
        device = next(self.parameters()).device
        orig_patience = self.patience
        max_neurons = int(_cfg_get(self.config, "max_neurons", 4096))
        delta = self.delta if delta is None else (0.0 if delta is None else float(delta))
        eps = 1e-12

        # capture constraint tensors for loss
        self.y_bus = constraints.get("y_bus")
        self.gen_bus_idx = constraints.get("gen_bus_idx")
        self.load_bus_idx = constraints.get("load_bus_idx")
        self.ineq_bounds = constraints.get("ineq")

        self.current_task = int(task_id)
        self.opt, self.scheduler = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Phase-local trackers (per-architecture best)
        best_snapshot = self.snapshot_state()
        best_val_mse = float("inf")

        # Accepted baseline (by validation MSE)
        accepted_snapshot = best_snapshot
        accepted_val_mse = float("inf")

        counter = orig_patience
        stop_outer = False
        just_expanded = False

        # remember starting widths for logs
        fc1_start = self.layers[0].out_features
        fc2_start = self.layers[-1].out_features

        # pre-expansion baseline for current series (updated only on ACCEPT)
        pre_exp_snapshot = accepted_snapshot
        pre_exp_val_mse = accepted_val_mse

        while not stop_outer:
            # ───────── inner loop (early stopping with penalty loss) ─────────
            for _ in range(max_epochs):
                self.train()
                for xb, yb, *_ in train_loader:
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

                # validate on pure MSE
                self.eval()
                with torch.no_grad():
                    val_mse, _ = self._evaluate_loader(val_loader, constraints, device)

                if val_mse < best_val_mse - eps:
                    best_val_mse = val_mse
                    best_snapshot = self.snapshot_state()
                    counter = orig_patience
                else:
                    counter -= 1

                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break
                if counter <= 0:
                    break

            if stop_outer:
                break

            # ───────── accept/retry/rollback for the previous expansion ─────────
            if just_expanded:
                if best_val_mse < pre_exp_val_mse - delta:
                    self._depth_failures = 0
                    accepted_snapshot = best_snapshot
                    accepted_val_mse = best_val_mse
                    just_expanded = False
                    logger.info("Width expansion accepted; resetting failure counter.")
                else:
                    self._depth_failures += 1
                    k, N = self._depth_failures, int(self.trials_depth)
                    if k < N:
                        logger.info(f"No val improvement after width expansion (trial {k}/{N}); trying another expansion.")
                        just_expanded = False
                    else:
                        logger.info(f"Expansions without improvement reached {N}; rolling back and stopping.")
                        self.restore_state(pre_exp_snapshot)
                        break

            # capacity guard
            if any(layer.out_features >= max_neurons for layer in self.layers):
                logger.info("Maximum capacity reached – stop expanding.")
                break

            # ───────── width-only expansion: widen all hidden layers by +ex_k ─────────
            pre_exp_snapshot = accepted_snapshot
            pre_exp_val_mse = accepted_val_mse
            self._widen_all_hidden(step=self.ex_k, max_neurons=max_neurons, device=device)
            logger.info(f"Expanded width by +{self.ex_k} neurons on each hidden layer.")

            # new optimiser/scheduler for new params
            self.opt, self.scheduler = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

            # reset per-architecture trackers for enlarged model
            best_val_mse = float("inf")
            counter = orig_patience
            just_expanded = True
            # continue → inner loop trains the larger net

        # end outer while — ensure we end with globally accepted best (by MSE)
        self.restore_state(accepted_snapshot)
        self._csv_append({"val_loss": accepted_val_mse}, fc1_start, fc2_start, len(train_loader), len(val_loader))
        return accepted_val_mse

    # ───────────────────────── width grow helper ─────────────────────────
    def _widen_all_hidden(self, step: int, max_neurons: int, device):
        prev_out = None
        for i, layer in enumerate(self.layers):
            new_in = layer.in_features if prev_out is None else prev_out
            new_out = min(max_neurons, layer.out_features + step)
            self.layers[i] = _resize_linear(layer, new_out, new_in).to(device)
            prev_out = new_out
        if self.head.in_features != prev_out:
            self.head = _resize_head(self.head, prev_out).to(device)

    # ───────────────────────── evaluation helper ─────────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()
        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device)), Yb.to(device)).item()
        val_loss = tot_mse / len(loader)

        # (optional) physics metrics for debugging
        X_all, Y_all = [], []
        with torch.no_grad():
            for Xb, *_ in loader:
                Xb = Xb.to(device)
                X_all.append(Xb)
                Y_all.append(self(Xb))
        X_val, Y_val = torch.cat(X_all), torch.cat(Y_all)
        P_res, Q_res = power_balance_residuals(
            Y_val,
            {
                "y_bus": constraints["y_bus"],
                "gen_bus_idx": constraints["gen_bus_idx"],
                "load_bus_idx": constraints["load_bus_idx"],
                "ineq": constraints["ineq"],
            },
        )
        ΔP, ΔQ, PG, QG, VM = mean_constraint_violation(
            Y_pred=Y_val,
            res_real=P_res,
            res_imag=Q_res,
            bounds=constraints["ineq"],
            num_gens=self.n_gen,
            num_buses=self.n_bus,
        )
        return val_loss, dict(ΔP=ΔP, ΔQ=ΔQ, PG=PG, QG=QG, VM=VM)

    # ───────────────────── snapshot helpers ─────────────────────
    def snapshot_state(self):
        return dict(
            state_dict=self.state_dict(),
            hidden_sizes=[l.out_features for l in self.layers],
        )

    def restore_state(self, snap):
        dev = next(self.parameters()).device
        target_sizes = snap["hidden_sizes"]

        # If depth differs, rebuild the stack to match snapshot
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
        log_dir = os.path.join("Results", f"{self.config.model}_{self.config.case_name}", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_task{self.current_task}_case-{self.config.case_name}.csv")
        minimal = {
            "task": self.current_task,
            "fc1_start": fc1_start,
            "fc2_start": fc2_start,
            "fc1_end": self.layers[0].out_features,
            "fc2_end": self.layers[-1].out_features,
            "val_loss": row["val_loss"],
        }
        df = pd.DataFrame([minimal])
        df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))


__all__ = ["PenaltyADP_DEN", "_resize_linear", "_resize_head"]
