"""
Penalty ADP-DEN (4-head) — Depth-Expand Only, with *patience on expansion*
──────────────────────────────────────────────────────────────────────────
This depth-only variant mirrors the non-penalty depth-only ADP_DEN and adds:

• Physics-informed penalty terms to the **training** loss (validation still MSE).
• A **patience-on-expansion** mechanism for depth growth:
    - Read `trials_depth = config.get('trials_depth', 5)` (fallback-safe).
    - Maintain `self._depth_failures` counter.
    - On plateau, start a depth-expansion *series*:
        • Append +1 hidden layer to `fc1` and each branch `fc2` (width unchanged).
        • Train to early stop; accept iff `best_val < pre_exp_val - delta`.
        • If accepted → reset counter, keep deeper arch, continue outer training.
        • If not accepted → increment counter.
            - If counter < trials_depth: **do not rollback**; try adding *another* layer immediately.
            - If counter ≥ trials_depth: rollback to the **pre-series** snapshot and stop expanding.
• Optimizer/scheduler are recreated after each architecture change.
• Capacity guard: respect `max_depth` exactly as the base does.

Acceptance threshold: uses `delta` (None → 0.0). No tiny eps.
"""

import logging
from types import SimpleNamespace
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

import pandas as pd
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals, mean_constraint_violation
from Dyn_DNN4OPF.models.adp_den_depth_only import ADP_DEN as _DepthOnlyBase

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


class PenaltyADP_DEN_4Head(_DepthOnlyBase):
    """4-head ADP with **depth-only expansion** and penalty loss + patience-on-expansion."""

    # ─────────────────────────────── init ───────────────────────────────
    def __init__(self, config: SimpleNamespace):
        super().__init__(config)
        # Penalty weights (fall back to config defaults if present)
        self.lambda_eq = getattr(config, "lambda_eq", 0.5)
        self.lambda_ineq = getattr(config, "lambda_ineq", 0.5)

        # Patience knob (dict-like .get with default; safe fallback) + counter
        try:
            self.trials_depth = config.get("trials_depth", 5)
        except AttributeError:
            self.trials_depth = getattr(config, "trials_depth", 5) if hasattr(config, "trials_depth") else 5
        self._depth_failures = 0

    # ───────────────────────────── loss ────────────────────────────────
    def _penalty_loss_batch(self, xb: torch.Tensor, yb: torch.Tensor, constraints: Dict[str, Any]) -> torch.Tensor:
        y_pred = self(xb)
        mse = F.mse_loss(y_pred, yb)

        # Equality (power balance) penalty — differentiable
        resP, resQ = power_balance_residuals(y_pred, constraints)  # (B, n_bus) each
        eq_pen = (resP.pow(2).mean() + resQ.pow(2).mean())

        # Inequality penalties on Pg, Qg, Vm (clamped hinge is differentiable)
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
            _viol(pg, p_min, p_max).mean()
            + _viol(qg, q_min, q_max).mean()
            + _viol(vm, v_min, v_max).mean()
        )

        return mse + self.lambda_eq * eq_pen + self.lambda_ineq * ineq_pen

    # ───────────── helpers to capture/shrink depths for rollback ─────────────
    def _get_depths(self) -> Dict[str, int]:
        return {
            "fc1": getattr(self.fc1, "num_layers", 1),
            **{f"{br}_fc2": getattr(getattr(self, f"{br}_fc2"), "num_layers", 1) for br in self.branches},
        }

    def _shrink_to_depths(self, depths: Dict[str, int]) -> None:
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
        """
        Same outer/inner control flow as the depth-only base, but after each plateau
        we run a **depth expansion series** with patience before deciding to stop.
        Only the TRAINING loss adds physics penalties; validation metric is MSE.
        """
        device = self.device
        patience_orig = self.patience
        max_depth = getattr(self.config, "max_depth", 16)
        # Acceptance threshold
        Δ = self.delta if delta is None else delta
        delta_thr = 0.0 if Δ is None else float(Δ)

        # fresh optimiser / scheduler for current architecture
        opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

        # Ensure wrappers exist (the base sets these in its first fit; repeat-safe)
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0
            if not hasattr(self.fc1, "append_depth"):
                # In case user instantiated but didn't run base fit yet
                from Dyn_DNN4OPF.models.adp_den_depth_only import StackedLinear
                self.fc1 = StackedLinear(self.fc1)
                for br in self.branches:
                    setattr(self, f"{br}_fc2", StackedLinear(getattr(self, f"{br}_fc2")))

        # Track last accepted baseline
        accepted_state = self.snapshot()
        accepted_val = float("inf")

        stop_all = False
        self._depth_failures = 0  # reset at task start

        while not stop_all:
            # ───── inner early-stopping loop (train current depth) ─────
            best_state = self.snapshot()
            best_val = float("inf")
            patience = patience_orig

            while patience > 0:
                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_all = True
                    break

                # train one epoch (penalty loss)
                self.train()
                for xb, yb, *_ in train_loader:
                    xb = xb.to(device, non_blocking=True)
                    yb = yb.to(device, non_blocking=True)
                    loss = self._penalty_loss_batch(xb, yb, constraints)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                try:
                    sched.step()
                except Exception:
                    pass

                # validate on pure MSE
                self.eval()
                val = 0.0
                with torch.no_grad():
                    for xb, yb, *_ in val_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
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
                # settle on best of the current depth
                self._restore(best_state)
                accepted_state = best_state
                accepted_val = min(accepted_val, best_val)
                break

            # Update accepted baseline for this plateau
            self._restore(best_state)
            accepted_state = best_state
            accepted_val = best_val

            # ───── Start a DEPTH expansion *series* with patience ─────
            pre_exp_state = accepted_state    # snapshot before the series
            pre_exp_val = accepted_val        # baseline val before the series
            pre_depths = self._get_depths()   # for structural rollback
            self._depth_failures = 0

            logger.info(
                f"Starting depth-expansion series from baseline val-MSE {pre_exp_val:.6f}; trials_depth={int(self.trials_depth)}."
            )

            # Ensure we start from the accepted baseline weights
            self._restore(pre_exp_state)

            while True:
                # Capacity guard: stop if any path already at depth limit
                if (
                    getattr(self.fc1, "num_layers", 1) >= max_depth
                    or any(getattr(getattr(self, f"{br}_fc2"), "num_layers", 1) >= max_depth for br in self.branches)
                ):
                    logger.info("Hit max_depth; stop depth expansions.")
                    # rollback to series baseline (no acceptance in this series)
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
                deep_best = float("inf")
                deep_state = self.snapshot()
                patience = patience_orig

                while patience > 0 and not stop_all:
                    self.global_epoch_count += 1
                    if self.global_epoch_count >= max_epochs:
                        stop_all = True
                        break

                    # train one epoch (penalty loss)
                    self.train()
                    for xb, yb, *_ in train_loader:
                        xb = xb.to(device, non_blocking=True)
                        yb = yb.to(device, non_blocking=True)
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
                            xb = xb.to(device, non_blocking=True)
                            yb = yb.to(device, non_blocking=True)
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

                # Handle global max_epochs while in deepened training
                if stop_all:
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

                # 4) Acceptance / patience-on-expansion logic
                if deep_best < pre_exp_val - delta_thr:
                    logger.info(
                        "Depth expansion accepted; counter reset. Improved %.6f → %.6f (delta=%.6f)",
                        pre_exp_val, deep_best, delta_thr,
                    )
                    self._restore(deep_state)
                    accepted_state = deep_state
                    accepted_val = deep_best
                    self._depth_failures = 0
                    break  # leave depth-series; continue outer loop to retrain at this depth
                else:
                    # FAIL: no improvement vs series baseline
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
                        # Keep the added capacity and attempt another +1 layer
                        self._restore(deep_state)
                        continue

            if stop_all:
                break

            # After an accepted depth expansion, re-init optimizer/scheduler and continue
            opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)
            continue

        # Final restore to last accepted model
        self._restore(accepted_state)
        return accepted_val

    # ───────────────────────────── eval helper ─────────────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()
        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device, non_blocking=True)), Yb.to(device, non_blocking=True)).item()
        val_loss = tot_mse / len(loader)

        with torch.no_grad():
            Y_pred = []
            Y_true = []
            for Xb, Yb, *_ in loader:
                Xb = Xb.to(device, non_blocking=True)
                Y_pred.append(self(Xb).detach().cpu())
                Y_true.append(Yb.detach().cpu())
            Y_pred = torch.cat(Y_pred, dim=0).to(device)
            Y_true = torch.cat(Y_true, dim=0).to(device)
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
