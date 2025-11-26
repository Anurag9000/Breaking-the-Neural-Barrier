"""
Penalty ADP-DEN (2-head: PQ + VM) — Depth-Expand Only, with *patience on expansion*
──────────────────────────────────────────────────────────────────────────
This mirrors the non-penalty 2-head depth-only model but replaces the training
criterion by a physics-informed penalty loss and adds a **patience-on-expansion**
mechanism for depth growth.

Patience rules (depth only):
• Read `trials_depth = config.get('trials_depth', 5)` (fallback-safe).
• Track `self._depth_failures`.
• At plateau, start a depth-expansion *series* from the accepted baseline:
  - Append +1 hidden layer to `fc1` and each branch `fc2` (width unchanged).
  - Train to early stop; accept iff `best_val < pre_exp_val - delta`.
  - If accepted → reset counter, keep deeper arch, continue outer training.
  - If not accepted → increment counter.
      · If counter < trials_depth: **do not rollback**; keep capacity and try another +1.
      · If counter ≥ trials_depth: rollback to the **pre-series** snapshot and stop expanding.

Other behaviors:
• Optimizer/scheduler are recreated after each architecture change.
• Capacity guard: respect `max_depth` exactly as the base does.
• Validation metric is MSE; penalties only affect the training loss.
• Acceptance uses `delta` (None→0.0), not a tiny eps.
"""

import logging
from types import SimpleNamespace
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals, mean_constraint_violation
from Dyn_DNN4OPF.models.adp_den_2head_depth_only import ADP_DEN_2Head as _DepthOnlyBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PenaltyADP_DEN_2Head(_DepthOnlyBase):
    """2-head ADP with **depth-only expansion** and penalty loss + patience-on-expansion.
    Branches: "pq" predicts [Pg,Qg], "vm" predicts [Va,Vm] (concatenated as PG|QG|VA|VM).
    """

    # ─────────────────────────────── init ───────────────────────────────
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        self.lambda_eq   = getattr(config, "lambda_eq", 0.5)
        self.lambda_ineq = getattr(config, "lambda_ineq", 0.5)
        # Patience knob (dict-like .get with default; safe fallback) + counter
        try:
            self.trials_depth = config.get('trials_depth', 5)
        except AttributeError:
            self.trials_depth = getattr(config, 'trials_depth', 5) if hasattr(config, 'trials_depth') else 5
        self._depth_failures = 0

    # ───────────────────────────── loss ────────────────────────────────
    def _penalty_loss_batch(self, xb: torch.Tensor, yb: torch.Tensor, constraints: Dict[str, Any]) -> torch.Tensor:
        y_pred = self(xb)
        mse = F.mse_loss(y_pred, yb)
        # Equality (power balance)
        resP, resQ = power_balance_residuals(y_pred, constraints)
        eq_pen = (resP.pow(2).mean() + resQ.pow(2).mean())
        # Inequality penalties on Pg, Qg, Vm
        n_g, n_b = self.n_gen, self.n_bus
        pg = y_pred[:, : n_g]
        qg = y_pred[:, n_g : 2 * n_g]
        vm = y_pred[:, 2 * n_g + n_b : 2 * n_g + 2 * n_b]
        b = constraints["ineq"]
        def _bt(k):
            v = b[k]
            return v if torch.is_tensor(v) else torch.as_tensor(v, device=y_pred.device, dtype=y_pred.dtype)
        p_min, p_max = _bt("p_min"), _bt("p_max")
        q_min, q_max = _bt("q_min"), _bt("q_max")
        v_min, v_max = _bt("v_min"), _bt("v_max")
        def _viol(x, lo, hi):
            return F.relu(x - hi) + F.relu(lo - x)
        ineq_pen = (
            _viol(pg, p_min, p_max).mean() +
            _viol(qg, q_min, q_max).mean() +
            _viol(vm, v_min, v_max).mean()
        )
        return mse + self.lambda_eq * eq_pen + self.lambda_ineq * ineq_pen

    # ───────────── helpers to capture/shrink depths for rollback ─────────────
    def _get_depths(self):
        return {
            "fc1": getattr(self.fc1, "num_layers", 1),
            **{f"{br}_fc2": getattr(getattr(self, f"{br}_fc2"), "num_layers", 1) for br in self.branches},
        }

    def _shrink_to_depths(self, depths):
        if hasattr(self.fc1, "shrink_to_depth"):
            self.fc1.shrink_to_depth(depths.get("fc1", 1))
        for br in self.branches:
            mod = getattr(self, f"{br}_fc2")
            if hasattr(mod, "shrink_to_depth"):
                mod.shrink_to_depth(depths.get(f"{br}_fc2", 1))

    # ────────────────────── training with patience on expansion ──────────────────────
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
        device        = self.device
        patience_orig = self.patience
        max_depth     = getattr(self.config, "max_depth", 16)
        # Acceptance threshold
        Δ = self.delta if delta is None else delta
        delta_thr = 0.0 if Δ is None else float(Δ)

        opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

        # Ensure wrappers exist (base sets on first fit; repeat-safe)
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0
            if not hasattr(self.fc1, "append_depth"):
                from Dyn_DNN4OPF.models.adp_den_2head_depth_only import StackedLinear
                self.fc1 = StackedLinear(self.fc1)
                for br in self.branches:
                    setattr(self, f"{br}_fc2", StackedLinear(getattr(self, f"{br}_fc2")))

        # Track last accepted baseline
        accepted_state = self.snapshot()
        accepted_val   = float('inf')

        stop_all = False
        self._depth_failures = 0  # reset at task start

        while not stop_all:
            # ───── inner early-stopping loop (train current depth) ─────
            best_state = self.snapshot()
            best_val   = float('inf')
            patience   = patience_orig

            while patience > 0:
                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_all = True
                    break
                # train (penalty loss)
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
                val = 0.0
                with torch.no_grad():
                    for xb, yb, *_ in val_loader:
                        xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                        val += F.mse_loss(self(xb), yb).item()
                val /= len(val_loader)
                logger.info(
                    f"[Task {self.current_task}] epoch {self.global_epoch_count:4d}  val-MSE {val:.6f}  patience {patience}"
                )
                if val < best_val - delta_thr:
                    best_val = val
                    best_state = self.snapshot()
                    patience = patience_orig
                else:
                    patience -= 1

            if stop_all:
                self._restore(best_state)
                accepted_state = best_state
                accepted_val = min(accepted_val, best_val)
                break

            # Update accepted baseline for this plateau
            self._restore(best_state)
            accepted_state = best_state
            accepted_val   = best_val

            # ───── Start a DEPTH expansion *series* with patience ─────
            pre_exp_state = accepted_state
            pre_exp_val   = accepted_val
            pre_depths    = self._get_depths()
            self._depth_failures = 0

            logger.info(
                f"Starting depth-expansion series from baseline val-MSE {pre_exp_val:.6f}; trials_depth={int(self.trials_depth)}."
            )

            # Ensure we start from the accepted baseline weights
            self._restore(pre_exp_state)

            while True:
                # Capacity guard
                if (
                    getattr(self.fc1, "num_layers", 1) >= max_depth
                    or any(getattr(getattr(self, f"{br}_fc2"), "num_layers", 1) >= max_depth for br in self.branches)
                ):
                    logger.info("Hit max_depth; stop depth expansions.")
                    # rollback to series baseline
                    self._shrink_to_depths(pre_depths)
                    self._restore(pre_exp_state)
                    accepted_state = pre_exp_state
                    accepted_val = pre_exp_val
                    stop_all = True
                    break

                # 1) deepen: append +1 hidden layer to fc1 and each branch fc2
                self.fc1.append_depth(1)
                for br in self.branches:
                    getattr(self, f"{br}_fc2").append_depth(1)
                logger.info("Deepened: added 1 layer to fc1 and each branch-fc2.")

                # 2) new optimiser/scheduler for deeper net
                opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

                # 3) train deepened model with early stopping
                deep_best  = float('inf')
                deep_state = self.snapshot()
                patience   = patience_orig

                while patience > 0 and not stop_all:
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
                    # validate
                    self.eval()
                    v = 0.0
                    with torch.no_grad():
                        for xb, yb, *_ in val_loader:
                            xb = xb.to(device, non_blocking=True); yb = yb.to(device, non_blocking=True)
                            v += F.mse_loss(self(xb), yb).item()
                    v /= len(val_loader)
                    logger.info(
                        f"[Task {self.current_task}] (deepened) epoch {self.global_epoch_count:4d}  val-MSE {v:.6f}  patience {patience}"
                    )
                    if v < deep_best - delta_thr:
                        deep_best = v
                        deep_state = self.snapshot()
                        patience = patience_orig
                    else:
                        patience -= 1

                if stop_all:
                    # choose better between deepened and pre-series baseline
                    if deep_best < pre_exp_val - delta_thr:
                        self._restore(deep_state)
                        accepted_state = deep_state
                        accepted_val = deep_best
                    else:
                        self._shrink_to_depths(pre_depths)
                        self._restore(pre_exp_state)
                        accepted_state = pre_exp_state
                        accepted_val = pre_exp_val
                    break

                # 4) Decide with *patience on expansion*
                if deep_best < pre_exp_val - delta_thr:
                    logger.info(
                        "Depth expansion accepted; counter reset. Improved %.6f → %.6f (delta=%.6f)",
                        pre_exp_val, deep_best, delta_thr,
                    )
                    self._restore(deep_state)
                    accepted_state = deep_state
                    accepted_val = deep_best
                    self._depth_failures = 0
                    break  # leave depth-series; continue outer loop at this depth
                else:
                    self._depth_failures += 1
                    if self._depth_failures >= int(self.trials_depth):
                        logger.info(
                            "Depth expansions without improvement reached %d; rolling back and stopping.",
                            int(self.trials_depth),
                        )
                        # Roll back architecture and weights to pre-series baseline
                        self._shrink_to_depths(pre_depths)
                        self._restore(pre_exp_state)
                        accepted_state = pre_exp_state
                        accepted_val = pre_exp_val
                        stop_all = True
                        break
                    else:
                        logger.info(
                            "No val improvement after depth expansion (trial %d/%d); trying another expansion.",
                            self._depth_failures, int(self.trials_depth),
                        )
                        # Keep added capacity; continue and try adding another layer
                        self._restore(deep_state)
                        continue

            if stop_all:
                break

            # After an accepted depth expansion, re-init optimizer/scheduler and continue
            opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)
            continue

        # Final restore
        self._restore(accepted_state)
        return accepted_val

    # (Evaluation helper identical to depth-only base)
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
            Y_val  = torch.cat(Y_val,  dim=0).to(device)
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
