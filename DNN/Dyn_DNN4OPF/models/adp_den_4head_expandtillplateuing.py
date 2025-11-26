"""
ADP-DEN  (mask-free version) with **expansion patience**
────────────────────────────────────────────────────────
• Inner loop  : early-stopping on validation MSE
• Outer loop  : add `ex_k` neurons to both hidden layers (shared fc1 + all branch fc2)
                accept expansion only if validation MSE improves by ≥ delta
• Patience    : allow up to N consecutive failed width expansions before rollback
                (N read via config.get('trials_width', 5))

Behavior
--------
- Maintain last *accepted* snapshot/score (accepted_snapshot / accepted_val_mse).
- Before each expansion series, capture pre-exp baselines from the accepted pair.
- After training the expanded model to plateau: accept iff best_val < pre_exp_val − delta.
- On failure within patience: keep the added capacity and immediately try another
  width expansion (sometimes two+ increments are needed to escape a basin).
- On patience exhaustion: rollback to pre-expansion snapshot and stop expanding.

All prior conventions preserved: capacity guards, optimizer/scheduler recreation
after each architecture change, device handling, logging, and CSV helpers.
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
# helpers
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


class ADP_DEN_4Head(ADPBase_4Head):
    # ─────────────────────────── init ───────────────────────────
    def __init__(self, config):
        super().__init__(config)
        self.delta = config.delta

        # patience (width-only for this model)
        self.trials_width: int = int(_cfg_get(self.config, 'trials_width', 5))
        self._width_failures: int = 0

        # optional output bounds
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
        Everything (inputs, params, temporaries) stays on GPU.
        The optional `BoundedAct` layer is applied last.
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
        Inner loop  : early-stopping on validation MSE.
        Outer loop  : progressive **width** expansions with patience.
        Accept an expansion only if `best_val < pre_exp_val - delta`.
        On failure within patience: keep width and try another expansion.
        On patience exhaustion: rollback to pre-exp snapshot and stop.
        Returns best accepted validation MSE for this task.
        """
        device        = self.device
        patience_orig = int(self.patience)
        ex_k          = int(self.ex_k)
        dtmp          = self.delta if delta is None else delta
        Δ             = 0.0 if dtmp is None else float(dtmp)
        max_neurons   = self.config.max_neurons

        # fresh optimiser / scheduler for *current* architecture
        opt, sched = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        # global epoch counter across all expansions
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # phase & global trackers
        best_state   = self.snapshot()          # phase-best (current arch)
        best_val_mse = float("inf")

        accepted_snapshot = best_state          # last globally accepted
        accepted_val_mse  = float("inf")

        stop_outer   = False
        just_expanded = False

        # pre-expansion baselines (updated only when an expansion is accepted)
        pre_exp_snapshot = accepted_snapshot
        pre_exp_val_mse  = accepted_val_mse

        # reset width-failure counter at task start
        self._width_failures = 0

        # ──────────────── OUTER plateau / expansion loop ────────────────
        while not stop_outer:

            # ───── inner early-stopping loop (train to plateau) ─────
            patience = patience_orig
            while patience > 0:
                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_outer = True
                    break

                # ---- train one epoch ----
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

                # ---- validate ----
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb, *_ in val_loader:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        val_loss += F.mse_loss(self(xb), yb).item()
                val_loss /= len(val_loader)

                logger.info(
                    f"[Task {self.current_task}] epoch {self.global_epoch_count:4d}  val-MSE {val_loss:.6f}  patience {patience}"
                )

                # early-stopping bookkeeping (strict improvement uses small eps)
                if val_loss < best_val_mse - 1e-12:
                    best_val_mse = val_loss
                    best_state   = self.snapshot()
                    patience     = patience_orig
                else:
                    patience -= 1

                # update globally accepted best as we go
                if best_val_mse < accepted_val_mse - 1e-12:
                    accepted_val_mse  = best_val_mse
                    accepted_snapshot = best_state

            # exhausted max_epochs inside inner loop?
            if stop_outer:
                break

            # always sit at phase-best weights
            self._restore(best_state)

            # ───── if we just trained an expanded net, decide accept/reject ─────
            if just_expanded:
                if best_val_mse < pre_exp_val_mse - Δ:
                    # ACCEPT width expansion
                    self._width_failures = 0
                    logger.info("Width expansion accepted; resetting failure counter.")
                    # move the pre-exp baselines forward to the newly accepted state
                    pre_exp_snapshot = accepted_snapshot
                    pre_exp_val_mse  = accepted_val_mse
                    just_expanded = False
                else:
                    # FAILED width expansion
                    self._width_failures += 1
                    k, N = self._width_failures, int(self.trials_width)
                    if k >= N:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back and stopping.")
                        # rollback to *pre-expansion* snapshot (last accepted)
                        self._restore(pre_exp_snapshot)
                        break
                    else:
                        logger.info(
                            f"No val improvement after width expansion (trial {k}/{N}); trying another expansion.")
                        # keep capacity and attempt another width step immediately
                        just_expanded = False

            # ───── plateau reached (current width) → attempt width expansion ─────
            # Roll back to accepted best of current width before expanding
            self._restore(accepted_snapshot)

            # capacity guard – stop if any relevant layer is already at limit
            if (
                self.fc1.out_features >= max_neurons
                or any(getattr(self, f"{br}_fc2").out_features >= max_neurons for br in self.branches)
            ):
                logger.info("Hit max_neurons; stop expanding.")
                break

            # ---------- enlarge shared fc1 and all branch fc2 ----------
            self.expand_layer(self.fc1, ex_k)
            for br in self.branches:
                self.expand_layer(getattr(self, f"{br}_fc2"), ex_k)
            logger.info("Expanded fc1 + branch-fc2 by %d neurons.", ex_k)

            # new optimiser / scheduler for the enlarged net
            opt, sched = get_optimizer_scheduler(
                self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
            )

            # reset inner-loop bookkeeping for the wider network
            best_val_mse = float("inf")
            patience     = patience_orig
            just_expanded = True
            # loop continues: next inner phase will evaluate the expanded net

        # end outer while — ensure we end with the best accepted weights
        self._restore(accepted_snapshot)
        return accepted_val_mse

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
            for Xb, *_ in loader:
                Xb = Xb.to(device, non_blocking=True)
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

        # Derive end widths without relying on a non-existent self.layers list
        fc1_end = self.fc1.out_features
        try:
            fc2_end = max(getattr(self, f"{br}_fc2").out_features for br in self.branches)
        except Exception:
            fc2_end = fc1_end  # fallback

        minimal = {
            "task":           self.current_task,
            "fc1_start":      fc1_start,                     # caller’s value
            "fc2_start":      fc2_start,                     # caller’s value
            "fc1_end":        fc1_end,                       # first hidden layer
            "fc2_end":        fc2_end,                       # last  hidden layer (max across branches)
            "val_loss":       row["val_loss"],
        }
        df = pd.DataFrame([minimal])
        df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
