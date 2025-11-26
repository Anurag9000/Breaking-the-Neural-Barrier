"""
ADP-DEN  (mask-free version) — with *patience on width expansion*
──────────────────────────────────────────────────────────────────
• Inner loop  : early-stopping on validation MSE
• Outer loop  : add `ex_k` neurons to both hidden layers (fc1, pq_fc2, vm_fc2)
                 Accept an expansion only if it beats the pre-expansion baseline by ≥ delta.
                 Allow up to N failed expansions (patience) before rolling back and stopping.

Patience knobs (read from config via .get with defaults):
    trials_width = config.get('trials_width', 5)
    delta        = config.delta (or passed to fit_task; None → 0.0)

Maintains:
    accepted_snapshot / accepted_val_mse – last *globally accepted* state/score
    pre_exp_snapshot / pre_exp_val_mse   – baseline captured right before a width-expansion *series*
    self._width_failures                 – consecutive failed width-expansion attempts

On failure within patience: keep capacity (no rollback) and attempt another expansion.
On patience exhausted: rollback architecture+weights to pre_exp_* and stop expanding.
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

# ════════════════════════════════════════════════════════════════════════
# basic linear-resize helpers
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
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def _resize_head(old: nn.Linear, new_in: int):
    """Resize only the input dimension of the output head."""
    return _resize_linear(old, old.out_features, new_in)


class ADP_DEN_2Head(ADPBase_2Head):
    def __init__(self, config):
        super().__init__(config)
        self.delta = config.delta

        # Patience knobs (dict-like .get with default; fallback if missing)
        try:
            self.trials_width = config.get("trials_width", 5)
        except AttributeError:
            self.trials_width = 5
        self._width_failures = 0  # counter for consecutive failed width expansions

        # --- override to two branches: PQ (Pg,Qg) and VM (Va,Vm) ---
        width = self.fc1.out_features
        device = self.device

        # PQ branch
        self.pq_fc2 = nn.Linear(width, width, bias=True, device=device)
        self.head_pq = nn.Linear(width, 2 * self.n_gen, bias=True, device=device)
        self.register_buffer(
            "pq_fc2_mask",
            torch.ones(self.pq_fc2.out_features, dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "pq_fc2_timestamp",
            torch.zeros(self.pq_fc2.out_features, dtype=torch.long, device=device),
        )

        # VM branch
        self.vm_fc2 = nn.Linear(width, width, bias=True, device=device)
        self.head_vm = nn.Linear(width, 2 * self.n_bus, bias=True, device=device)
        self.register_buffer(
            "vm_fc2_mask",
            torch.ones(self.vm_fc2.out_features, dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "vm_fc2_timestamp",
            torch.zeros(self.vm_fc2.out_features, dtype=torch.long, device=device),
        )

        # override branch list for super().forward
        self.branches = ["pq", "vm"]

        # optional bounded activation as before
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(
                config.bounds_low, config.bounds_high, config.mask
            )
            self.bound_layer.apply_bounds.fill_(False)
        else:
            self.bound_layer = nn.Identity()
            self.bound_layer.apply_bounds = torch.tensor(False, dtype=torch.bool)

    def forward(self, x):
        """
        Pipeline:
            shared fc1  → branch-specific fc2  → heads (PQ then VM)
        Everything stays on GPU; gating/masks applied in ADPBase.
        The optional `BoundedAct` layer is applied last.
        """
        return self.bound_layer(super().forward(x))

    # ─────────────────────────── helpers for (un)doing expansion ───────────────────────────
    def _get_sizes(self) -> dict:
        """Capture current hidden sizes (for reverting failed expansions)."""
        return {
            "fc1": self.fc1.out_features,
            "pq_fc2": self.pq_fc2.out_features,
            "vm_fc2": self.vm_fc2.out_features,
        }

    def _shrink_to(self, sizes: dict) -> None:
        """
        Shrink layers back to `sizes` and adjust heads' input dims accordingly.
        Uses weight-preserving linear resize.
        """
        # fc1
        if self.fc1.out_features != sizes["fc1"]:
            self.fc1 = _resize_linear(self.fc1, sizes["fc1"], self.fc1.in_features).to(self.device)

        # pq branch
        if self.pq_fc2.out_features != sizes["pq_fc2"]:
            self.pq_fc2 = _resize_linear(self.pq_fc2, sizes["pq_fc2"], self.pq_fc2.in_features).to(self.device)
        if self.head_pq.in_features != self.pq_fc2.out_features:
            self.head_pq = _resize_head(self.head_pq, self.pq_fc2.out_features).to(self.device)

        # vm branch
        if self.vm_fc2.out_features != sizes["vm_fc2"]:
            self.vm_fc2 = _resize_linear(self.vm_fc2, sizes["vm_fc2"], self.vm_fc2.in_features).to(self.device)
        if self.head_vm.in_features != self.vm_fc2.out_features:
            self.head_vm = _resize_head(self.head_vm, self.vm_fc2.out_features).to(self.device)

    # ────────────────────────────────────────────────────────────────────
    # Training with *patience on expansion* (width-only model)
    # ────────────────────────────────────────────────────────────────────
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
        Outer loop  : when the inner loop plateaus, repeatedly attempt **width expansion**
                      (add `ex_k` neurons to fc1 and to each branch fc2). Use patience-on-expansion
                      before rolling back and halting growth.

        Acceptance rule: accept if best_val_mse < pre_exp_val_mse - delta (delta None→0.0).

        Returns
        -------
        float  –  best validation MSE obtained for this task (last accepted baseline).
        """
        device = self.device
        patience_orig = self.patience
        ex_k = self.ex_k
        max_neurons = getattr(self.config, "max_neurons", None)
        # delta precedence: explicit arg > self.delta; None→0.0
        if delta is None:
            delta = 0.0 if self.delta is None else float(self.delta)
        else:
            delta = float(delta)

        # Reset width failure counter at task start
        self._width_failures = 0

        # fresh optimiser / scheduler for *current* architecture
        opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

        # global epoch counter across all expansions
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        stop_all = False

        # First: train current architecture to plateau (inner early-stop)
        accepted_snapshot = self.snapshot()
        accepted_val_mse = float("inf")

        while True:
            # ───── inner early-stopping loop for the *current* architecture ─────
            best_state = self.snapshot()
            best_val_mse = float("inf")
            patience = patience_orig

            while patience > 0:
                self.global_epoch_count += 1
                if self.global_epoch_count >= max_epochs:
                    stop_all = True
                    break

                # train one epoch
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

                # validate
                self.eval()
                val_loss = 0.0
                with torch.no_grad():
                    for xb, yb, *_ in val_loader:
                        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                        val_loss += F.mse_loss(self(xb), yb).item()
                val_loss /= len(val_loader)

                logger.info(
                    f"[Task {self.current_task}] epoch {self.global_epoch_count:4d}  "
                    f"val-MSE {val_loss:.6f}  patience {patience}"
                )

                if val_loss < best_val_mse - delta:
                    best_val_mse = val_loss
                    best_state = self.snapshot()
                    patience = patience_orig
                else:
                    patience -= 1

            if stop_all:
                # Restore best known weights and exit
                self._restore(best_state)
                accepted_snapshot = best_state
                accepted_val_mse = min(accepted_val_mse, best_val_mse)
                break

            # Lock in best for current width as the latest accepted baseline
            self._restore(best_state)
            accepted_snapshot = best_state
            accepted_val_mse = best_val_mse

            # ───── Plateau reached → start a *series* of width expansions with patience ─────
            pre_exp_snapshot = accepted_snapshot  # baseline before the series
            pre_exp_val_mse = accepted_val_mse
            pre_sizes = self._get_sizes()

            logger.info(
                f"Starting width-expansion series from baseline val-MSE {pre_exp_val_mse:.6f}; "
                f"patience(trials_width)={self.trials_width}."
            )

            # Ensure we start expansions from the accepted baseline weights
            self._restore(pre_exp_snapshot)

            # Attempt multiple expansions until accept or patience exhausts
            while True:
                # Capacity guard
                if max_neurons is not None:
                    if (
                        self.fc1.out_features >= max_neurons
                        or any(getattr(self, f"{br}_fc2").out_features >= max_neurons for br in self.branches)
                        or (self.fc1.out_features + ex_k > max_neurons)
                        or any(getattr(self, f"{br}_fc2").out_features + ex_k > max_neurons for br in self.branches)
                    ):
                        logger.info("Hit max_neurons guard; stopping width expansions.")
                        # If we haven't accepted within this series, rollback to baseline
                        self._shrink_to(pre_sizes)
                        self._restore(pre_exp_snapshot)
                        accepted_snapshot = pre_exp_snapshot
                        accepted_val_mse = pre_exp_val_mse
                        stop_all = True
                        break

                # 1) Widen all hidden layers by +ex_k
                #    Keep using project helper if available (ADPBase.expand_layer)
                try:
                    self.expand_layer(self.fc1, ex_k)
                    for br in self.branches:
                        self.expand_layer(getattr(self, f"{br}_fc2"), ex_k)
                except AttributeError:
                    # Fallback: manual _resize_linear
                    self.fc1 = _resize_linear(self.fc1, self.fc1.out_features + ex_k, self.fc1.in_features).to(device)
                    for br in self.branches:
                        fc2 = getattr(self, f"{br}_fc2")
                        setattr(self, f"{br}_fc2", _resize_linear(fc2, fc2.out_features + ex_k, fc2.in_features).to(device))
                logger.info("Expanded fc1 + branch-fc2 by %d neurons.", ex_k)

                # 2) (re)size heads to match widened fc2 inputs
                self.head_pq = _resize_head(self.head_pq, self.pq_fc2.out_features).to(device)
                self.head_vm = _resize_head(self.head_vm, self.vm_fc2.out_features).to(device)

                # 3) New optimizer/scheduler for the changed parameter set
                opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

                # 4) Train the widened model with inner early-stopping
                widen_best_mse = float("inf")
                widen_best_state = self.snapshot()
                patience = patience_orig

                while patience > 0 and not stop_all:
                    self.global_epoch_count += 1
                    if self.global_epoch_count >= max_epochs:
                        stop_all = True
                        break

                    # train one epoch
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

                    # validate
                    self.eval()
                    val_loss = 0.0
                    with torch.no_grad():
                        for xb, yb, *_ in val_loader:
                            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                            val_loss += F.mse_loss(self(xb), yb).item()
                    val_loss /= len(val_loader)

                    logger.info(
                        f"[Task {self.current_task}] (widened) epoch {self.global_epoch_count:4d}  "
                        f"val-MSE {val_loss:.6f}  patience {patience}"
                    )

                    if val_loss < widen_best_mse - delta:
                        widen_best_mse = val_loss
                        widen_best_state = self.snapshot()
                        patience = patience_orig
                    else:
                        patience -= 1

                # If max_epochs hit during widened training, decide vs. baseline and exit
                if stop_all:
                    if widen_best_mse < pre_exp_val_mse - delta:
                        self._restore(widen_best_state)
                        accepted_snapshot = widen_best_state
                        accepted_val_mse = widen_best_mse
                    else:
                        # rollback to baseline of the series
                        self._shrink_to(pre_sizes)
                        self._restore(pre_exp_snapshot)
                        accepted_snapshot = pre_exp_snapshot
                        accepted_val_mse = pre_exp_val_mse
                    break

                # 5) Post-expansion decision with *patience on expansion*
                if widen_best_mse < pre_exp_val_mse - delta:
                    # ACCEPT: keep, reset counter, refresh baseline, leave series
                    logger.info(
                        "Width expansion accepted; resetting failure counter. "
                        f"Improved val-MSE {pre_exp_val_mse:.6f} → {widen_best_mse:.6f} (delta={delta})."
                    )
                    self._restore(widen_best_state)
                    accepted_snapshot = widen_best_state
                    accepted_val_mse = widen_best_mse
                    self._width_failures = 0
                    break  # leave expansion *series*; go train this arch again
                else:
                    # FAIL: increment counter; decide retry vs rollback
                    self._width_failures += 1
                    if self._width_failures >= self.trials_width:
                        logger.info(
                            f"Width expansions without improvement reached {self.trials_width}; "
                            "rolling back and stopping."
                        )
                        # Roll back architecture and weights to series baseline
                        self._shrink_to(pre_sizes)
                        self._restore(pre_exp_snapshot)
                        accepted_snapshot = pre_exp_snapshot
                        accepted_val_mse = pre_exp_val_mse
                        stop_all = True
                        break
                    else:
                        logger.info(
                            f"No val improvement after width expansion (trial {self._width_failures}/"
                            f"{self.trials_width}); trying another expansion."
                        )
                        # Keep added capacity; try another widening (append another +ex_k)
                        self._restore(widen_best_state)
                        continue

            # End expansion series
            if stop_all:
                break

            # After an accepted expansion, continue outer loop to re-train to plateau at new width
            opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)
            continue

        # restore accepted best weights before returning
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
        vm = Y_val[:, 2 * self.n_gen + self.n_bus : 2 * (self.n_gen + self.n_bus)]
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

        minimal = {
            "task": self.current_task,
            "fc1_start": fc1_start,  # caller’s value
            "fc2_start": fc2_start,  # caller’s value
            "fc1_end": self.fc1.out_features,  # first hidden layer
            "fc2_end": max(self.pq_fc2.out_features, self.vm_fc2.out_features),  # last hidden width across heads
            "val_loss": row["val_loss"],
        }
        df = pd.DataFrame([minimal])
        df.to_csv(log_file, mode="a", index=False, header=not os.path.exists(log_file))
