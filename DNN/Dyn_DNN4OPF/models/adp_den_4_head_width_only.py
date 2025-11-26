"""
ADP-DEN (4-head) — width-only expansion with patience
─────────────────────────────────────────────────────
• Mirrors `adp_den_4head_depth_only.py` exactly; only the *expansion step* differs.
• Inner loop: early stopping on MSE. Outer loop: on plateau, **widen fc1 and every branch fc2** by +ex_k neurons.
• Acceptance rule: **physics-first** — accept only if (ΔP+ΔQ) strictly decreases.
• Patience on expansion: allow up to `trials_depth` failures, then rollback to last accepted.
• Capacity guard: stop if any hidden width ≥ `max_neurons`.

Config keys respected: delta, trials_depth, max_neurons, patience, ex_k (new; default 16)
"""

import os
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from models.adp_base_4head import ADPBase_4Head
from Dyn_DNN4OPF.utils.constraint_losses import (
    mean_constraint_violation,
    power_balance_residuals,
)
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ─────────────────────────────────────────────────────────────────────────────
# Linear resize utilities (copy semantics from single-head depth-only helpers)
# ─────────────────────────────────────────────────────────────────────────────

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


def _cfg_get(cfg, key: str, default):
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


class ADP_DEN_4Head(ADPBase_4Head):
    def __init__(self, config):
        super().__init__(config)
        self.delta = config.delta
        self.trials_depth: int = int(_cfg_get(config, "trials_depth", 10))
        self._depth_failures: int = 0
        self.ex_k: int = int(_cfg_get(config, "ex_k", 16))

        # optional bounded activation as before
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(
                config.bounds_low, config.bounds_high, config.mask
            )
            self.bound_layer.apply_bounds.fill_(False)   # train/val
        else:
            self.bound_layer = nn.Identity()
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

    def forward(self, x):
        """
        Pipeline:
            shared fc1  → branch-specific fc2  → branch heads
        Optional `BoundedAct` is applied last.
        """
        return self.bound_layer(super().forward(x))

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
        Early-stopping on MSE; on plateau, attempt **width** expansion of fc1 and all branch fc2 (+ex_k neurons).
        Acceptance: strictly better physics → (ΔP+ΔQ) decreases.
        Patience: allow up to `trials_depth` consecutive failed expansions before rollback.
        Returns the accepted model's MSE.
        """
        device        = self.device
        patience_orig = self.patience
        max_neurons   = int(_cfg_get(self.config, "max_neurons", 4096))
        eps           = 1e-12

        # fresh optimiser / scheduler for *current* architecture
        opt, sched = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Phase-local trackers
        best_snapshot = self.snapshot()  # deep copy (provided by base)
        best_val_mse  = float("inf")

        # Seed accepted baseline from current state (physics-first sem.)
        self._restore(best_snapshot)
        _vl, phys = self._evaluate_loader(val_loader, constraints, device)
        accepted_snapshot = best_snapshot
        accepted_val_mse  = _vl
        accepted_csum     = phys["ΔP"] + phys["ΔQ"]

        # Pre-expansion baseline placeholders
        pre_exp_snapshot = None
        pre_exp_csum     = None

        counter       = patience_orig
        stop_outer    = False
        just_expanded = False

        # widths for CSV logging
        fc1_start = self.fc1.out_features
        try:
            fc2_start = max(getattr(self, f"{br}_fc2").out_features for br in self.branches)
        except Exception:
            fc2_start = self.fc1.out_features

        while not stop_outer:
            # ───────── inner loop (MSE early-stopping) ─────────
            for _ in range(max_epochs):
                self.train()
                for xb, yb, *_ in train_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    loss = F.mse_loss(self(xb), yb)
                    loss.backward()
                    opt.step()

                if sched is not None:
                    try:
                        sched.step()
                    except Exception:
                        pass

                self.eval()
                with torch.no_grad():
                    val_mse, _ = self._evaluate_loader(val_loader, constraints, device)

                if val_mse < best_val_mse - eps:
                    best_val_mse = val_mse
                    best_snapshot = self.snapshot()
                    # also track physics at phase-best for *possible* acceptance
                    self._restore(best_snapshot)
                    _vl, phys = self._evaluate_loader(val_loader, constraints, device)
                    best_csum = phys["ΔP"] + phys["ΔQ"]
                    counter = patience_orig
                else:
                    counter -= 1

                self.global_epoch_count += 1
                logger.info(
                    f"[Task {self.current_task}] epoch {self.global_epoch_count:4d}  val-MSE {val_mse:.6f}  patience {counter}"
                )

                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break

                if counter <= 0:
                    break

            if stop_outer:
                break

            # ───────── accept/retry/rollback after an expansion ─────────
            if just_expanded:
                if best_csum < pre_exp_csum - eps:
                    # ACCEPT → reset failure counter, update pre-exp baseline
                    self._depth_failures = 0
                    accepted_snapshot = best_snapshot
                    accepted_val_mse  = min(accepted_val_mse, best_val_mse)
                    accepted_csum     = best_csum
                    logger.info("Width expansion accepted; resetting failure counter.")
                    just_expanded = False
                else:
                    # FAIL within patience
                    self._depth_failures += 1
                    k, N = self._depth_failures, int(self.trials_depth)
                    if k < N:
                        logger.info(
                            f"No physics improvement after width expansion (trial {k}/{N}); trying another expansion.")
                        just_expanded = False
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back and stopping.")
                        # rollback to pre-expansion baseline and stop expanding
                        self._restore(pre_exp_snapshot)
                        break

            # ───────── capacity guard ─────────
            if (
                self.fc1.out_features >= max_neurons or
                any(getattr(self, f"{br}_fc2").out_features >= max_neurons for br in self.branches)
            ):
                logger.info("Maximum width reached – stop expanding.")
                break

            # ───────── perform *width* expansion on fc1 + each branch fc2 ─────────
            pre_exp_snapshot = accepted_snapshot
            pre_exp_csum     = accepted_csum

            self._widen_shared_and_branches(step=self.ex_k, max_neurons=max_neurons, device=device)
            logger.info(f"Expanded width by +{self.ex_k} on fc1 and each branch fc2.")

            # new optimiser/scheduler for new params
            opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

            # reset phase trackers for enlarged model
            best_val_mse  = float("inf")
            best_csum     = float("inf")
            counter       = patience_orig
            just_expanded = True
            # loop continues → inner training on bigger net

        # restore final accepted snapshot (physics-first)
        self._restore(accepted_snapshot)
        # CSV log for the task
        self._csv_append({"val_loss": accepted_val_mse}, fc1_start, fc2_start, len(train_loader), len(val_loader))
        return accepted_val_mse

    # ──────────────────────────────────────────────────────────────
    # Width grow helper: fc1 + every branch fc2 (+ resize heads)
    # ──────────────────────────────────────────────────────────────
    def _widen_shared_and_branches(self, step: int, max_neurons: int, device):
        # 1) widen shared fc1
        new_fc1_out = min(max_neurons, self.fc1.out_features + step)
        self.fc1 = _resize_linear(self.fc1, new_fc1_out, self.fc1.in_features).to(device)

        # 2) for each branch: resize fc2.in to fc1.out; widen fc2.out by step; resize head input
        for br in self.branches:
            fc2 = getattr(self, f"{br}_fc2")
            new_fc2_in  = self.fc1.out_features
            new_fc2_out = min(max_neurons, fc2.out_features + step)
            fc2 = _resize_linear(fc2, new_fc2_out, new_fc2_in).to(device)
            setattr(self, f"{br}_fc2", fc2)

            # try common head naming conventions: head_{br} preferred
            head_attr = None
            if hasattr(self, f"head_{br}"):
                head_attr = f"head_{br}"
            elif hasattr(self, f"{br}_head"):
                head_attr = f"{br}_head"
            if head_attr is not None:
                setattr(self, head_attr, _resize_head(getattr(self, head_attr), new_fc2_out).to(device))

    # ───────────────────────── evaluation ─────────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()

        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device, non_blocking=True)), Yb.to(device, non_blocking=True)).item()
        val_loss = tot_mse / len(loader)

        # physics metrics (same as before)
        with torch.no_grad():
            Y_pred, Y_val = [], []
            for Xb, Yb, *_ in loader:
                Xb = Xb.to(device, non_blocking=True)
                Y_pred.append(self(Xb).detach().cpu())
                Y_val.append(Yb.detach().cpu())
            Y_pred = torch.cat(Y_pred, dim=0)
            Y_val  = torch.cat(Y_val, dim=0)

            P_res, Q_res = power_balance_residuals(
                Y_pred,
                y_bus=constraints["y_bus"],
                gen_bus_idx=constraints["gen_bus_idx"],
                load_bus_idx=constraints["load_bus_idx"],
                num_buses=self.n_bus,
            )
            ΔP, ΔQ, PG, QG, VM = mean_constraint_violation(
                P_res=P_res, Q_res=Q_res,
                y_bounds=constraints["ineq"],
                num_gens=self.n_gen,
                num_buses=self.n_bus,
            )

        return val_loss, dict(ΔP=ΔP, ΔQ=ΔQ, PG=PG, QG=QG, VM=VM)

    # ────────────────────── _csv_append (log first & last) ─────────────────────
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

        try:
            fc1_end = self.fc1.out_features
            fc2_end = max(getattr(self, f"{br}_fc2").out_features for br in self.branches)
        except Exception:
            fc1_end = fc2_start
            fc2_end = fc2_start

        minimal = {
            "task":           self.current_task,
            "fc1_start":      fc1_start,
            "fc2_start":      fc2_start,
            "fc1_end":        fc1_end,
            "fc2_end":        fc2_end,
            "val_loss":       row["val_loss"],
        }
        df = pd.DataFrame([minimal])
        df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
