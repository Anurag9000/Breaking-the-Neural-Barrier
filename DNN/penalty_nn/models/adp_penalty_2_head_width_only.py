"""
Penalty ADP-DEN (2-head: PQ + VM) — Width-Expand Only, with *patience on expansion*
──────────────────────────────────────────────────────────────────────────
Mirrors `adp_penalty_2head_depth_only.py`, but uses **width growth** (fc1, pq_fc2, vm_fc2 widened by +ex_k)
instead of depth appends. Training adds physics penalties; validation uses MSE. Acceptance and patience semantics
are unchanged: accept if `best_val < pre_exp_val − delta`; allow up to `trials_depth` failures otherwise rollback.
"""

import os
import logging
from types import SimpleNamespace
from typing import Any, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals, mean_constraint_violation
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
from models.adp_base_2head import ADPBase_2Head

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


class PenaltyADP_DEN_2Head(ADPBase_2Head):
    """2-head ADP with **width-only** expansion and penalty training + patience-on-expansion.
    Branches: "pq" predicts [Pg,Qg], "vm" predicts [Va,Vm] (concatenated as PG|QG|VA|VM).
    """

    # ─────────────────────────────── init ───────────────────────────────
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.lambda_eq = getattr(config, "lambda_eq", 0.5)
        self.lambda_ineq = getattr(config, "lambda_ineq", 0.5)
        try:
            self.trials_depth = config.get('trials_depth', 10)
        except AttributeError:
            self.trials_depth = getattr(config, 'trials_depth', 10) if hasattr(config, 'trials_depth') else 10
        self._depth_failures = 0
        self.ex_k = int(_cfg_get(config, "ex_k", 16))

        # optional bounded activation
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(config.bounds_low, config.bounds_high, config.mask)
            self.bound_layer.apply_bounds.fill_(False)
        else:
            self.bound_layer = nn.Identity()
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        pq = F.relu(self.pq_fc2(x))
        vm = F.relu(self.vm_fc2(x))
        pq = self.head_pq(pq)
        vm = self.head_vm(vm)
        out = torch.cat([pq, vm], dim=-1)
        return self.bound_layer(out)

    # ───────────────────────────── loss ────────────────────────────────
    def _penalty_loss_batch(self, xb: torch.Tensor, yb: torch.Tensor, constraints: Dict[str, Any]) -> torch.Tensor:
        y_pred = self(xb)
        mse = F.mse_loss(y_pred, yb)
        # Equality (power balance)
        resP, resQ = power_balance_residuals(y_pred, constraints)
        eq_pen = (resP.abs().mean() + resQ.abs().mean())
        # Inequality penalties on Pg, Qg, Vm
        n_g, n_b = self.n_gen, self.n_bus
        pg = y_pred[:, : n_g]
        qg = y_pred[:, n_g : 2 * n_g]
        vm = y_pred[:, 2 * n_g + n_b : 2 * (n_g + n_b)]
        _, _, PG, QG, VM = mean_constraint_violation(
            Y_pred=y_pred,
            res_real=resP,
            res_imag=resQ,
            bounds=constraints["ineq"],
            num_gens=self.n_gen,
            num_buses=self.n_bus,
        )
        ineq_pen = (PG + QG + VM) / 3.0
        return mse + self.lambda_eq * eq_pen + self.lambda_ineq * ineq_pen

    # ─────────────────────────── fit (plateau → *widen*) ───────────────────────────
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
        device = self.device
        patience_orig = self.patience
        Δ = self.delta if delta is None else delta
        delta_thr = 0.0 if Δ is None else float(Δ)
        max_neurons = int(_cfg_get(self.config, "max_neurons", 4096))

        opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Seed accepted baseline
        self.eval()
        with torch.no_grad():
            accepted_state = self.snapshot()
            accepted_val, _ = self._evaluate_loader(val_loader, constraints, device)

        stop_all = False
        self._depth_failures = 0

        # widths for CSV logging
        fc1_start = self.fc1.out_features
        fc2_start = max(self.pq_fc2.out_features, self.vm_fc2.out_features)

        while not stop_all:
            # ───── inner early-stopping on current architecture ─────
            best_state = self.snapshot()
            best_val = float("inf")
            patience = patience_orig

            while patience > 0:
                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_all = True
                    break
                # train (penalty)
                self.train()
                for xb, yb, *_ in train_loader:
                    xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                    loss = self._penalty_loss_batch(xb, yb, constraints)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                try:
                    sched.step()
                except Exception:
                    pass

                # validate (MSE)
                self.eval()
                v, _ = self._evaluate_loader(val_loader, constraints, device)
                if v < best_val - 1e-12:
                    best_val = v
                    best_state = self.snapshot()
                    patience = patience_orig
                else:
                    patience -= 1
                if patience <= 0:
                    break

            # accepted baseline update
            if best_val < accepted_val - delta_thr:
                accepted_val = best_val
                accepted_state = best_state

            # ───── Start a WIDTH expansion series with patience ─────
            pre_exp_state = accepted_state
            pre_exp_val = accepted_val
            self._depth_failures = 0
            self._restore(pre_exp_state)

            while True:
                # capacity guard
                if (
                    self.fc1.out_features >= max_neurons or
                    self.pq_fc2.out_features >= max_neurons or
                    self.vm_fc2.out_features >= max_neurons
                ):
                    logger.info("Hit max width; stop width expansions.")
                    self._restore(pre_exp_state)
                    accepted_state = pre_exp_state
                    accepted_val = pre_exp_val
                    stop_all = True
                    break

                # 1) widen shared + branches
                self._widen_shared_and_branches(step=self.ex_k, max_neurons=max_neurons, device=device)

                # 2) fresh optimiser
                opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

                # 3) train widened model
                deep_best = float('inf')
                deep_state = self.snapshot()
                patience = patience_orig

                while patience > 0 and not stop_all:
                    self.global_epoch_count += 1
                    if self.global_epoch_count >= max_epochs:
                        stop_all = True
                        break
                    # train
                    self.train()
                    for xb, yb, *_ in train_loader:
                        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                        loss = self._penalty_loss_batch(xb, yb, constraints)
                        opt.zero_grad(set_to_none=True)
                        loss.backward()
                        opt.step()
                    try:
                        sched.step()
                    except Exception:
                        pass
                    # validate
                    self.eval()
                    v, _ = self._evaluate_loader(val_loader, constraints, device)
                    if v < deep_best - 1e-12:
                        deep_best = v
                        deep_state = self.snapshot()
                        patience = patience_orig
                    else:
                        patience -= 1

                # global cap handling
                if stop_all:
                    if deep_best < pre_exp_val - delta_thr:
                        self._restore(deep_state)
                        accepted_state = deep_state
                        accepted_val = deep_best
                    else:
                        self._restore(pre_exp_state)
                        accepted_state = pre_exp_state
                        accepted_val = pre_exp_val
                    break

                # acceptance / patience-on-expansion
                if deep_best < pre_exp_val - delta_thr:
                    logger.info(
                        "Width expansion accepted; counter reset. Improved %.6f → %.6f (delta=%.6f)",
                        pre_exp_val, deep_best, delta_thr,
                    )
                    self._restore(deep_state)
                    accepted_state = deep_state
                    accepted_val = deep_best
                    self._depth_failures = 0
                    pre_exp_state = accepted_state
                    pre_exp_val = accepted_val
                else:
                    self._depth_failures += 1
                    k, N = self._depth_failures, int(self.trials_depth)
                    if k < N:
                        logger.info(
                            f"No MSE improvement after width expansion (trial {k}/{N}); trying another expansion.")
                        continue
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back and stopping.")
                        self._restore(pre_exp_state)
                        accepted_state = pre_exp_state
                        accepted_val = pre_exp_val
                        break

            if stop_all:
                break

        # restore final accepted state
        self._restore(accepted_state)
        # CSV log
        self._csv_append({"val_loss": accepted_val}, fc1_start, fc2_start, len(train_loader), len(val_loader))
        return accepted_val

    # ──────────────────────────────────────────────────────────────
    # Width grow helper: fc1 + pq_fc2 + vm_fc2 (+ resize heads)
    # ──────────────────────────────────────────────────────────────
    def _widen_shared_and_branches(self, step: int, max_neurons: int, device):
        # shared fc1
        new_fc1_out = min(max_neurons, self.fc1.out_features + step)
        self.fc1 = _resize_linear(self.fc1, new_fc1_out, self.fc1.in_features).to(device)
        # pq branch
        new_pq_in = self.fc1.out_features
        new_pq_out = min(max_neurons, self.pq_fc2.out_features + step)
        self.pq_fc2 = _resize_linear(self.pq_fc2, new_pq_out, new_pq_in).to(device)
        self.head_pq = _resize_head(self.head_pq, new_pq_out).to(device)
        # vm branch
        new_vm_in = self.fc1.out_features
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
            Y_pred, Y_val = [], []
            for Xb, Yb, *_ in loader:
                Xb = Xb.to(device, non_blocking=True)
                Y_pred.append(self(Xb).detach().cpu())
                Y_val.append(Yb.detach().cpu())
            Y_pred = torch.cat(Y_pred, dim=0).to(device)
            Y_val = torch.cat(Y_val, dim=0).to(device)
            resP, resQ = power_balance_residuals(Y_pred, constraints)
        ΔP, ΔQ, PG, QG, VM = mean_constraint_violation(
            Y_pred=Y_pred,
            res_real=resP,
            res_imag=resQ,
            bounds=constraints["ineq"],
            num_gens=self.n_gen,
            num_buses=self.n_bus,
        )
        return val_loss, dict(ΔP=ΔP, ΔQ=ΔQ, PG=PG, QG=QG, VM=VM)

    # ────────────────────── _csv_append (log first & last) ─────────────────────
    def _csv_append(self, row, fc1_start, fc2_start, n_tr, n_val):
        log_dir = os.path.join("Results", f"{self.config.model}_{self.config.case_name}", "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, f"log_task{self.current_task}_case-{self.config.case_name}.csv")
        minimal = {
            "task": self.current_task,
            "fc1_start": fc1_start,
            "fc2_start": fc2_start,
            "fc1_end": self.fc1.out_features,
            "fc2_end": max(self.pq_fc2.out_features, self.vm_fc2.out_features),
            "val_loss": row["val_loss"],
        }
        df = pd.DataFrame([minimal])
        df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
