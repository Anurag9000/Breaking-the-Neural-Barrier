"""
ADP-DEN  (mask-free version) with **expansion patience**
────────────────────────────────────────────────────────
• Inner loop  : early-stopping on validation MSE
• Outer loop  : add one hidden layer to fc1 and all branch fc2 (keep width)
                accept expansion only if ΔP+ΔQ improves (by a strict eps)
• Patience    : allow up to `trials_depth` consecutive failed expansions
                before rolling back to the last accepted snapshot.

Config keys used
----------------
- delta            : early-stopping improvement margin (MSE)
- trials_depth     : max consecutive failed depth expansions (default=5)
- max_depth        : depth guard for each stacked path (default=16)

Snapshots tracked
-----------------
- accepted_snapshot / accepted_csum : last globally accepted state (by ΔP+ΔQ)
- pre_exp_snapshot  / pre_exp_csum  : baseline captured just before the
                                      current *series* of depth expansions
                                      (updated when an expansion is accepted)
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

# ════════════════════════════════════════════════════════════════════════
# Depth-expansion helper: stack Linear layers while keeping width
# ════════════════════════════════════════════════════════════════════════
class StackedLinear(nn.Module):
    def __init__(self, first: nn.Linear):
        super().__init__()
        self.layers = nn.ModuleList([first])

    @property
    def in_features(self) -> int:  # type: ignore[override]
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:  # type: ignore[override]
        return self.layers[-1].out_features

    @property
    def num_layers(self) -> int:
        return sum(isinstance(m, nn.Linear) for m in self.layers)

    def append_depth(self, n: int = 1, device=None) -> None:
        if device is None:
            device = self.layers[-1].weight.device
        width = self.out_features
        for _ in range(n):
            self.layers.append(nn.Linear(width, width, bias=True, device=device))

    def shrink_to_depth(self, depth: int) -> None:
        kept: list[nn.Module] = []
        count = 0
        for m in self.layers:
            if isinstance(m, nn.Linear):
                count += 1
            if count <= depth:
                kept.append(m)
        self.layers = nn.ModuleList(kept)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        linear_layers = [m for m in self.layers if isinstance(m, nn.Linear)]
        for i, m in enumerate(linear_layers):
            x = m(x)
            if i < len(linear_layers) - 1:
                x = F.relu(x)
        return x


def _cfg_get(cfg, key: str, default):
    """Dictionary-style get with attribute fallback."""
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


class ADP_DEN_4Head(ADPBase_4Head):
    def __init__(self, config):
        super().__init__(config)
        self.delta = config.delta
        self.trials_depth: int = int(_cfg_get(config, "trials_depth", 5))
        self._depth_failures: int = 0

        # optional bounded activation as before
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(
                config.bounds_low, config.bounds_high, config.mask
            )
            self.bound_layer.apply_bounds.fill_(False)   # train/val
        else:
            self.bound_layer = nn.Identity()
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

        # Wrap shared/branch layers so we can grow *depth* while keeping width
        if not isinstance(self.fc1, StackedLinear):
            self.fc1 = StackedLinear(self.fc1)
        for br in self.branches:
            mod = getattr(self, f"{br}_fc2")
            if not isinstance(mod, StackedLinear):
                setattr(self, f"{br}_fc2", StackedLinear(mod))

    def forward(self, x):
        """
        Pipeline:
            shared fc1  → branch-specific fc2  → branch heads
        Everything stays on GPU. Optional `BoundedAct` is applied last.
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
        Early-stopping on MSE; on plateau, attempt **depth** expansion of fc1
        and all branch fc2 by appending one same-width layer each.

        Expansion acceptance: strictly better physics → (ΔP+ΔQ) decreases.
        Patience: allow up to `trials_depth` consecutive failed expansions,
        keeping added layers between attempts; on exhaustion, rollback to
        the last *accepted* snapshot and stop expanding.
        """
        device        = self.device
        patience_orig = self.patience
        max_depth     = int(_cfg_get(self.config, "max_depth", 16))
        eps           = 1e-12

        # fresh optimiser / scheduler for *current* architecture
        opt, sched = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        # global epoch counter across all expansions
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Phase-local trackers
        best_snapshot = self.snapshot()  # deep copy
        best_val_mse  = float("inf")
        # compute physics at current state to seed accepted baseline
        _vl, physics0 = self._evaluate_loader(val_loader, constraints, device)
        best_csum = physics0["ΔP"] + physics0["ΔQ"]

        # Global accepted best by physics
        accepted_snapshot = best_snapshot
        accepted_csum     = best_csum
        accepted_val_mse  = _vl

        # pre-expansion baseline for the current *series*
        pre_exp_snapshot = accepted_snapshot
        pre_exp_csum     = accepted_csum

        counter       = patience_orig
        stop_outer    = False
        just_expanded = False
        self._depth_failures = 0  # reset at task start

        while not stop_outer:
            # ───────── inner loop (early stopping) ─────────
            for _ in range(max_epochs):
                self.train()
                for xb, yb, *_ in train_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    loss = F.mse_loss(self(xb), yb)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    opt.step()
                try:
                    sched.step()
                except Exception:
                    pass

                # validation (pure MSE)
                self.eval()
                val_sum = 0.0
                with torch.no_grad():
                    for xb, yb, *_ in val_loader:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        val_sum += F.mse_loss(self(xb), yb).item()
                val_mse = val_sum / max(1, len(val_loader))

                improved = val_mse < best_val_mse - (0.0 if self.delta is None else float(self.delta))
                if improved:
                    best_val_mse = val_mse
                    best_snapshot = self.snapshot()
                    # physics at phase-best
                    self._restore(best_snapshot)
                    _vl, phys = self._evaluate_loader(val_loader, constraints, device)
                    best_csum = phys["ΔP"] + phys["ΔQ"]
                    counter = patience_orig
                else:
                    counter -= 1

                self.global_epoch_count += 1

                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break
                if counter == 0:
                    break  # plateau → consider expansion / decision

            # ensure we are at phase-best weights
            self._restore(best_snapshot)

            # update globally accepted best by physics
            _vl, phys_now = self._evaluate_loader(val_loader, constraints, device)
            cur_csum = phys_now["ΔP"] + phys_now["ΔQ"]
            if cur_csum < accepted_csum - eps:
                accepted_csum = cur_csum
                accepted_snapshot = self.snapshot()
                accepted_val_mse = _vl

            if stop_outer:
                break

            # ───────── accept/retry/rollback after an expansion ─────────
            if just_expanded:
                if best_csum < pre_exp_csum - eps:
                    # ACCEPT → reset failure counter, update pre-exp baseline
                    self._depth_failures = 0
                    pre_exp_snapshot = accepted_snapshot
                    pre_exp_csum     = accepted_csum
                    logger.info("Depth expansion accepted; resetting failure counter.")
                    just_expanded = False
                else:
                    # FAIL within patience
                    self._depth_failures += 1
                    k, N = self._depth_failures, int(self.trials_depth)
                    if k >= N:
                        logger.info(
                            f"Depth expansions without improvement reached {N}; rolling back and stopping.")
                        self._restore(pre_exp_snapshot)
                        break
                    else:
                        logger.info(
                            f"No ΔP+ΔQ improvement after expansion (trial {k}/{N}); trying another depth expansion.")
                        just_expanded = False  # keep capacity; attempt another expansion below

            # capacity/depth guard
            if (
                getattr(self.fc1, "num_layers", 1) >= max_depth
                or any(getattr(getattr(self, f"{br}_fc2"), "num_layers", 1) >= max_depth for br in self.branches)
            ):
                logger.info("Maximum depth reached – stop expanding.")
                break

            # ───────── actually append +1 layer (depth-only) ─────────
            # set current accepted as pre-exp baseline for the new attempt
            pre_exp_snapshot = accepted_snapshot
            pre_exp_csum     = accepted_csum

            self.fc1.append_depth(1, device=device)
            for br in self.branches:
                getattr(self, f"{br}_fc2").append_depth(1, device=device)
            logger.info("Expanded depth by +1 layer on fc1 and each branch fc2 (width preserved).")

            # new optimiser/scheduler for new params
            opt, sched = get_optimizer_scheduler(
                self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
            )

            # reset inner-loop bookkeeping for the enlarged model
            best_val_mse  = float("inf")
            counter       = patience_orig
            just_expanded = True
            continue

        # end outer while — ensure we end with the best physics snapshot
        self._restore(accepted_snapshot)
        return accepted_val_mse

    # ───────────────────────── evaluation helper ─────────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()
        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device, non_blocking=True)), Yb.to(device, non_blocking=True)).item()
        val_loss = tot_mse / len(loader)

        # collect preds for constraint metrics
        X_all, Y_all = [], []
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                Xb = Xb.to(device, non_blocking=True)
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
            pg,
            qg,
            pd,
            qd,
            vm,
            va,
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

        try:
            fc1_end = self.fc1.out_features
            fc2_end = max(getattr(self, f"{br}_fc2").out_features for br in self.branches)
        except Exception:
            fc1_end = fc2_start  # fallback
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
