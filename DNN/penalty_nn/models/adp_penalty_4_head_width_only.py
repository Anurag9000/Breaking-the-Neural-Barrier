"""
Penalty-ADP-DEN (4-head) — width-only expansion with outer patience
───────────────────────────────────────────────────────────────────
Mirrors `adp_penalty_4head_depth_only.py` semantics, but replaces *depth append* with
*width increase* of the shared trunk and each branch's fc2 by +ex_k neurons.

• Train loss (per batch):  L = λ_mse·MSE + λ_eq·mean(|ΔP|+|ΔQ|) + λ_ineq·mean(hinge(PG,QG,VM))
• Validation metric:       pure MSE (no penalties)
• Inner loop:              early-stopping on val MSE (patience = self.patience)
• Outer loop:              on plateau, widen (fc1 & all branch fc2). Accept iff
                           best_val < pre_exp_val − delta. Otherwise retry until
                           `trials_depth` failures, then rollback to pre-expansion baseline.

Defaults: delta comes from config (None→0.0). Outer patience `trials_depth` defaults to 10.
"""

from __future__ import annotations
import logging
from typing import Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct
from models.adp_base_4head import ADPBase_4Head

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
        new.bias[:r]       = old.bias[:r]
    return new


def _resize_head(old: nn.Linear, new_in: int):
    return _resize_linear(old, old.out_features, new_in)


def _cfg_get(cfg, key: str, default):
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


class PenaltyADP_DEN_4Head(ADPBase_4Head):
    def __init__(self, config):
        super().__init__(config)
        # thresholds / patience knobs
        self.delta = _cfg_get(config, "delta", 0.0)
        self.trials_depth: int = int(_cfg_get(config, "trials_depth", 10))  # outer patience (expansion failures)
        self._depth_failures: int = 0
        self.ex_k: int = int(_cfg_get(config, "ex_k", 16))

        # penalty weights (defaults mirror depth-only penalty files)
        self.lambda_mse: float = float(_cfg_get(config, "lambda_mse", 1.0))
        self.lambda_eq: float = float(_cfg_get(config, "lambda_eq", 1.0))
        self.lambda_ineq: float = float(_cfg_get(config, "lambda_ineq", 1.0))

        # optional bounded activation (disabled for train/val)
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(config.bounds_low, config.bounds_high, config.mask)
            self.bound_layer.apply_bounds.fill_(False)
        else:
            self.bound_layer = nn.Identity()
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

    def forward(self, x):
        return self.bound_layer(super().forward(x))

    # ───────────────────────── penalty loss (train) ─────────────────────────
    def _penalty_loss_batch(self, xb: torch.Tensor, yb: torch.Tensor, constraints: Dict[str, Any]) -> torch.Tensor:
        y_pred = self.forward(xb)
        base_mse = F.mse_loss(y_pred, yb)

        # slice outputs assuming order: [PG | QG | VA | VM]
        n_g, n_b = self.n_gen, self.n_bus
        pg = y_pred[:, : n_g]
        qg = y_pred[:, n_g : 2 * n_g]
        va = y_pred[:, 2 * n_g : 2 * n_g + n_b]
        vm = y_pred[:, 2 * n_g + n_b : 2 * (n_g + n_b)]

        # equality residuals (AC power balance)
        resP, resQ = power_balance_residuals(
            y_pred,
            y_bus=constraints["y_bus"],
            gen_bus_idx=constraints["gen_bus_idx"],
            load_bus_idx=constraints["load_bus_idx"],
            num_buses=self.n_bus,
        )
        eq_term = resP.abs().mean() + resQ.abs().mean()

        # inequality hinge (box bounds)
        ineq = constraints["ineq"]
        p_min, p_max = ineq["p_min"], ineq["p_max"]
        q_min, q_max = ineq["q_min"], ineq["q_max"]
        v_min, v_max = ineq["v_min"], ineq["v_max"]
        pg_v = (F.relu(pg - p_max) + F.relu(p_min - pg)).mean()
        qg_v = (F.relu(qg - q_max) + F.relu(q_min - qg)).mean()
        vm_v = (F.relu(vm - v_max) + F.relu(v_min - vm)).mean()
        ineq_term = (pg_v + qg_v + vm_v) / 3.0

        return self.lambda_mse * base_mse + self.lambda_eq * eq_term + self.lambda_ineq * ineq_term

    # ───────────────────────── training (plateau → *widen*) ─────────────────────────
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
        max_neurons = int(_cfg_get(self.config, "max_neurons", 4096))
        delta_thr = 0.0 if (self.delta if delta is None else delta) is None else float(self.delta if delta is None else delta)
        eps = 1e-12

        # fresh optimiser / scheduler for current architecture
        opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # Phase-local trackers (per-architecture best)
        best_snapshot = self.snapshot()
        best_val_mse = float("inf")

        # Accepted baseline (by validation MSE)
        self._restore(best_snapshot)
        v0, _ = self._evaluate_loader(val_loader, constraints, device)
        accepted_snapshot = best_snapshot
        accepted_val_mse = v0

        # pre-expansion baseline
        pre_exp_snapshot = None
        pre_exp_val_mse = None

        counter = patience_orig
        stop_outer = False
        just_expanded = False

        while not stop_outer:
            # ───────── inner loop (penalty train, MSE val) ─────────
            for _ in range(max_epochs):
                self.train()
                for xb, yb, *_ in train_loader:
                    xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                    opt.zero_grad(set_to_none=True)
                    loss = self._penalty_loss_batch(xb, yb, constraints)
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
                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break
                if counter <= 0:
                    break

            if stop_outer:
                break

            # ───────── accept/retry/rollback for previous expansion ─────────
            if just_expanded:
                if best_val_mse < pre_exp_val_mse - delta_thr:
                    self._depth_failures = 0
                    accepted_snapshot = best_snapshot
                    accepted_val_mse = best_val_mse
                    just_expanded = False
                    logger.info("Width expansion accepted; resetting failure counter.")
                else:
                    self._depth_failures += 1
                    k, N = self._depth_failures, int(self.trials_depth)
                    if k < N:
                        logger.info(f"No MSE improvement after width expansion (trial {k}/{N}); trying another expansion.")
                        just_expanded = False
                    else:
                        logger.info(f"Expansions without improvement reached {N}; rolling back and stopping.")
                        self._restore(pre_exp_snapshot)
                        break

            # capacity guard
            if (
                self.fc1.out_features >= max_neurons or
                any(getattr(self, f"{br}_fc2").out_features >= max_neurons for br in self.branches)
            ):
                logger.info("Maximum width reached – stop expanding.")
                break

            # ───────── width-only expansion: widen fc1 and all branch fc2 ─────────
            pre_exp_snapshot = accepted_snapshot
            pre_exp_val_mse = accepted_val_mse
            self._widen_shared_and_branches(step=self.ex_k, max_neurons=max_neurons, device=device)

            # new optimiser/scheduler for new params
            opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

            # reset per-architecture trackers
            best_val_mse = float("inf")
            counter = patience_orig
            just_expanded = True
            # loop continues → inner training on bigger net

        # end: restore accepted best (by MSE)
        self._restore(accepted_snapshot)
        return accepted_val_mse

    # ──────────────────────────────────────────────────────────────
    # Width grow helper: fc1 + every branch fc2 (+ resize heads)
    # ──────────────────────────────────────────────────────────────
    def _widen_shared_and_branches(self, step: int, max_neurons: int, device):
        # 1) shared fc1
        new_fc1_out = min(max_neurons, self.fc1.out_features + step)
        self.fc1 = _resize_linear(self.fc1, new_fc1_out, self.fc1.in_features).to(device)

        # 2) each branch fc2 (+ resize head inputs)
        for br in self.branches:
            fc2 = getattr(self, f"{br}_fc2")
            new_fc2_in = self.fc1.out_features
            new_fc2_out = min(max_neurons, fc2.out_features + step)
            fc2 = _resize_linear(fc2, new_fc2_out, new_fc2_in).to(device)
            setattr(self, f"{br}_fc2", fc2)

            # head attribute name conventions
            if hasattr(self, f"head_{br}"):
                setattr(self, f"head_{br}", _resize_head(getattr(self, f"head_{br}"), new_fc2_out).to(device))
            elif hasattr(self, f"{br}_head"):
                setattr(self, f"{br}_head", _resize_head(getattr(self, f"{br}_head"), new_fc2_out).to(device))

    # ───────────────────────── evaluation ─────────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()
        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device, non_blocking=True)), Yb.to(device, non_blocking=True)).item()
        val_loss = tot_mse / len(loader)

        # (optional) physics diagnostics can be computed similarly if desired
        return val_loss, {}


__all__ = ["PenaltyADP_DEN_4Head"]
