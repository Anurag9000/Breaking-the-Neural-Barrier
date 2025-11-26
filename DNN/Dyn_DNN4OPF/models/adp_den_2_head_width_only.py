"""
ADP-DEN (2-head: PQ + VM) — width-only expansion with patience
───────────────────────────────────────────────────────────────
• Mirrors `adp_den_2head_depth_only.py` exactly; only the *expansion step* differs.
• Inner loop: early-stopping on MSE (patience = self.patience).
• Outer loop: on plateau, **widen fc1, pq_fc2, vm_fc2** by +ex_k neurons; resize heads.
• Acceptance (as in depth 2-head): **MSE-first** — accept if best_val_mse < pre_exp_val_mse − delta.
• Expansion patience: allow up to `trials_depth` failures; then rollback architecture & weights to pre-expansion baseline.
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
from models.adp_base_2head import ADPBase_2Head
from Dyn_DNN4OPF.utils.constraint_losses import (
    mean_constraint_violation,
    power_balance_residuals,
)
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

# ─────────────────────────────────────────────────────────────────────────────
# Linear resize utilities (width-only growth)
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


class ADP_DEN_2Head(ADPBase_2Head):
    def __init__(self, config):
        super().__init__(config)
        self.delta = _cfg_get(config, "delta", 0.0)
        self.trials_depth: int = int(_cfg_get(config, "trials_depth", 10))
        self._depth_failures: int = 0
        self.ex_k: int = int(_cfg_get(config, "ex_k", 16))

        # optional bounded activation as in depth file
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(
                config.bounds_low, config.bounds_high, config.mask
            )
            self.bound_layer.apply_bounds.fill_(False)   # train/val disabled
        else:
            self.bound_layer = nn.Identity()
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

    def forward(self, x):
        # shared trunk
        x = F.relu(self.fc1(x))
        # branch heads
        pq = F.relu(self.pq_fc2(x))
        vm = F.relu(self.vm_fc2(x))
        pq = self.head_pq(pq)
        vm = self.head_vm(vm)
        out = torch.cat([pq, vm], dim=-1)
        return self.bound_layer(out)

    # ───────────────────────── training (plateau → widen with patience) ─────────────────────────
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
        Train with early stopping; on plateau, attempt **width-only** expansion of fc1, pq_fc2, vm_fc2 (+ex_k neurons).
        Acceptance (MSE-first): accept iff best_val_mse < pre_exp_val_mse − delta. Allow up to `trials_depth` failures
        before rolling back to the pre-expansion baseline (both architecture and weights). Returns final accepted MSE.
        """
        device        = self.device
        patience_orig = self.patience
        max_neurons   = int(_cfg_get(self.config, "max_neurons", 4096))
        # mirror depth 2-head precedence (arg → self.delta → 0.0)
        delta_eff = 0.0 if delta is None else float(delta if self.delta is None else self.delta)
        eps        = 1e-12

        # fresh optimiser / scheduler
        opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Phase-local trackers (per-architecture best)
        best_snapshot = self.snapshot()
        best_val_mse  = float("inf")

        # Accepted baseline (by MSE rule)
        accepted_snapshot = best_snapshot
        accepted_val_mse  = float("inf")

        # Pre-exp baselines for current expansion series
        pre_exp_snapshot = None
        pre_exp_val_mse  = None
        pre_depths       = None  # not used here; kept to mirror structure

        # widths for CSV logging
        fc1_start = self.fc1.out_features
        fc2_start = max(self.pq_fc2.out_features, self.vm_fc2.out_features)

        counter       = patience_orig
        stop_outer    = False
        just_expanded = False

        while not stop_outer:
            # ───────── inner loop (MSE early stopping) ─────────
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

            # ───────── acceptance for the *previous* expansion ─────────
            if just_expanded:
                if best_val_mse < pre_exp_val_mse - delta_eff:
                    # success → reset failure counter and update accepted baseline
                    self._depth_failures = 0
                    accepted_snapshot = best_snapshot
                    accepted_val_mse  = best_val_mse
                    logger.info("Width expansion accepted; resetting failure counter.")
                    just_expanded = False
                else:
                    self._depth_failures += 1
                    k, N = self._depth_failures, int(self.trials_depth)
                    if k < N:
                        logger.info(
                            f"No MSE improvement after width expansion (trial {k}/{N}); trying another expansion.")
                        just_expanded = False
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back and stopping.")
                        # rollback to pre-expansion baseline (arch+weights)
                        self._restore(pre_exp_snapshot)
                        break

            # ───────── capacity guard ─────────
            if (
                self.fc1.out_features >= max_neurons or
                self.pq_fc2.out_features >= max_neurons or
                self.vm_fc2.out_features >= max_neurons
            ):
                logger.info("Maximum width reached – stop expanding.")
                break

            # ───────── perform *width* expansion on fc1 + both branches ─────────
            pre_exp_snapshot = accepted_snapshot
            pre_exp_val_mse  = accepted_val_mse

            self._widen_shared_and_branches(step=self.ex_k, max_neurons=max_neurons, device=device)
            logger.info(f"Expanded width by +{self.ex_k} on fc1, pq_fc2, vm_fc2.")

            # new optimiser/scheduler for new params
            opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

            # reset per-architecture trackers
            best_val_mse  = float("inf")
            counter       = patience_orig
            just_expanded = True
            # loop continues → inner training on bigger net

        # restore final accepted snapshot
        self._restore(accepted_snapshot)
        # CSV log
        self._csv_append({"val_loss": accepted_val_mse}, fc1_start, fc2_start, len(train_loader), len(val_loader))
        return accepted_val_mse

    # ──────────────────────────────────────────────────────────────
    # Width grow helper: fc1 + pq_fc2 + vm_fc2 (+ resize heads)
    # ──────────────────────────────────────────────────────────────
    def _widen_shared_and_branches(self, step: int, max_neurons: int, device):
        # shared fc1
        new_fc1_out = min(max_neurons, self.fc1.out_features + step)
        self.fc1 = _resize_linear(self.fc1, new_fc1_out, self.fc1.in_features).to(device)

        # pq branch
        new_pq_in  = self.fc1.out_features
        new_pq_out = min(max_neurons, self.pq_fc2.out_features + step)
        self.pq_fc2 = _resize_linear(self.pq_fc2, new_pq_out, new_pq_in).to(device)
        self.head_pq = _resize_head(self.head_pq, new_pq_out).to(device)

        # vm branch
        new_vm_in  = self.fc1.out_features
        new_vm_out = min(max_neurons, self.vm_fc2.out_features + step)
        self.vm_fc2 = _resize_linear(self.vm_fc2, new_vm_out, new_vm_in).to(device)
        self.head_vm = _resize_head(self.head_vm, new_vm_out).to(device)

    # ───────────────────────── evaluation ─────────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()

        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device, non_blocking=True)), Yb.to(device, non_blocking=True)).item()
        val_loss = tot_mse / len(loader)

        with torch.no_grad():
            preds = []
            for Xb, *_ in loader:
                Xb = Xb.to(device, non_blocking=True)
                preds.append(self(Xb).detach().cpu())
            Y_pred = torch.cat(preds, dim=0)

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

        minimal = {
            "task":           self.current_task,
            "fc1_start":      fc1_start,
            "fc2_start":      fc2_start,
            "fc1_end":        self.fc1.out_features,
            "fc2_end":        max(self.pq_fc2.out_features, self.vm_fc2.out_features),
            "val_loss":       row["val_loss"],
        }
        df = pd.DataFrame([minimal])
        df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
