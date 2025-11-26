"""
ADP-DEN  (mask-free version) — with *patience on expansion*
────────────────────────────────────────────────────────────
• Inner loop   : early-stopping on validation MSE
• Outer loop   : depth expansion (append one hidden layer to fc1 and both branch fc2)
                 Accept an expansion only if it beats the pre-expansion baseline by ≥ delta.
                 Allow up to N failed expansions (patience) before rolling back and stopping.

Patience knobs (read from config via .get with defaults):
    trials_depth = config.get('trials_depth', 5)
    delta        = config.delta (or passed to fit_task; None → 0.0)

This implementation maintains:
    accepted_snapshot / accepted_val_mse – last *globally accepted* state/score
    pre_exp_snapshot / pre_exp_val_mse   – baseline captured right before an expansion *series*
    self._depth_failures                 – consecutive failed expansions counter

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
# basic linear-resize helpers (unchanged)
# ════════════════════════════════════════════════════════════════════════

def _resize_linear(old: nn.Linear, new_out: int, new_in: int | None = None):
    """Return a new Linear(new_in, new_out) with overlapping weights copied."""
    if new_in is None:
        new_in = old.in_features
    new = nn.Linear(new_in, new_out, bias=True).to(old.weight.device)
    with torch.no_grad():
        # copy intersection
        oh, ow = old.weight.shape
        nh, nw = new.weight.shape
        h = min(oh, nh)
        w = min(ow, nw)
        new.weight[:h, :w].copy_(old.weight[:h, :w])
        if old.bias is not None and new.bias is not None:
            new.bias[:h].copy_(old.bias[:h])
    return new


def _resize_head(old: nn.Linear, new_in: int):
    return _resize_linear(old, old.out_features, new_in)


# ════════════════════════════════════════════════════════════════════════
# Depth-expansion helper: StackedLinear keeps width, grows depth by appending
# ════════════════════════════════════════════════════════════════════════
class StackedLinear(nn.Module):
    def __init__(self, first: nn.Linear):
        super().__init__()
        self.layers = nn.ModuleList([first])

    @property
    def in_features(self) -> int:
        return self.layers[0].in_features

    @property
    def out_features(self) -> int:
        return self.layers[-1].out_features

    @property
    def num_layers(self) -> int:
        return len([m for m in self.layers if isinstance(m, nn.Linear)])

    def append_depth(self, n: int = 1, device=None):
        if device is None:
            device = self.layers[-1].weight.device
        width = self.out_features
        for _ in range(n):
            self.layers.append(nn.Linear(width, width, bias=True, device=device))

    def shrink_to_depth(self, depth: int):
        # keep only the first "depth" Linear layers
        kept = []
        count = 0
        for m in self.layers:
            if isinstance(m, nn.Linear):
                count += 1
            if count <= depth:
                kept.append(m)
        self.layers = nn.ModuleList(kept)

    def forward(self, x):
        # apply ReLU between internal hidden layers, but not after the last one
        for i, m in enumerate([m for m in self.layers if isinstance(m, nn.Linear)]):
            x = m(x)
            if i < self.num_layers - 1:
                x = F.relu(x)
        return x


class ADP_DEN_2Head(ADPBase_2Head):
    def __init__(self, config):
        super().__init__(config)
        self.delta = config.delta

        # Patience knobs (dict-like .get with default; fallback if missing)
        try:
            self.trials_depth = config.get("trials_depth", 5)
        except AttributeError:
            # Fallback if config is namespace-like; spec prefers .get
            self.trials_depth = 5
        self._depth_failures = 0

        # --- override to two branches: PQ (Pg,Qg) and VM (Va,Vm) ---
        width = self.fc1.out_features
        device = self.device
        # Wrap shared fc1 to allow depth growth while keeping width
        if not isinstance(self.fc1, StackedLinear):
            self.fc1 = StackedLinear(self.fc1)

        # PQ branch: combines Pg & Qg
        self.pq_fc2 = StackedLinear(nn.Linear(width, width, bias=True, device=device))
        self.head_pq = nn.Linear(width, 2 * self.n_gen, bias=True, device=device)
        self.register_buffer(
            "pq_fc2_mask",
            torch.ones(self.pq_fc2.out_features, dtype=torch.bool, device=device),
        )
        self.register_buffer(
            "pq_fc2_timestamp",
            torch.zeros(self.pq_fc2.out_features, dtype=torch.long, device=device),
        )

        # VM branch: combines Va & Vm
        self.vm_fc2 = StackedLinear(nn.Linear(width, width, bias=True, device=device))
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
        # delegate to ADPBase, which now uses branches=["pq","vm"]
        return self.bound_layer(super().forward(x))

    # ─────────────────────────── helpers for (un)doing expansion ───────────────────────────
    def _get_sizes(self) -> dict:
        """Capture current *depths* (number of Linear layers) to allow reverting failed expansions."""
        return {
            "fc1_depth": self.fc1.num_layers if isinstance(self.fc1, StackedLinear) else 1,
            "pq_fc2_depth": self.pq_fc2.num_layers if isinstance(self.pq_fc2, StackedLinear) else 1,
            "vm_fc2_depth": self.vm_fc2.num_layers if isinstance(self.vm_fc2, StackedLinear) else 1,
        }

    def _shrink_to(self, sizes: dict) -> None:
        """
        Shrink depth back to recorded values (width stays constant).
        """
        if isinstance(self.fc1, StackedLinear):
            self.fc1.shrink_to_depth(sizes["fc1_depth"])
        if isinstance(self.pq_fc2, StackedLinear):
            self.pq_fc2.shrink_to_depth(sizes["pq_fc2_depth"])
        if isinstance(self.vm_fc2, StackedLinear):
            self.vm_fc2.shrink_to_depth(sizes["vm_fc2_depth"])

    # ────────────────────────────────────────────────────────────────────
    # Training with *patience on expansion* (depth-only model)
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
        Outer loop  : when the inner loop plateaus, repeatedly attempt **depth expansion**
                      (append one hidden layer to fc1 and to each branch fc2). Use patience
                      on expansion (self.trials_depth) before rolling back and halting growth.

        Acceptance (unchanged in spirit): accept if best_val_mse < pre_exp_val_mse - delta.

        Returns
        -------
        float  –  best validation MSE obtained for this task (i.e., last accepted baseline).
        """
        device = self.device
        patience_orig = self.patience
        max_depth = getattr(self.config, "max_depth", 16)
        delta = 0.0 if delta is None else float(delta if self.delta is None else self.delta)
        # NOTE: If caller passes delta explicitly, prefer that; else use self.delta; None→0.0
        # Reset expansion failure counter at task start
        self._depth_failures = 0

        # fresh optimiser / scheduler for *current* architecture
        opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

        # global epoch counter across all expansions
        if not hasattr(self, "global_epoch_count"):
            self.global_epoch_count = 0

        # First: train current architecture to plateau (inner early-stop)
        accepted_snapshot = self.snapshot()
        accepted_val_mse = float("inf")

        stop_all = False
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
                except Exception:  # keep existing conventions
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

            # Exhausted or hit max-epochs?
            if stop_all:
                # Restore best known weights and exit
                self._restore(best_state)
                accepted_snapshot = best_state
                accepted_val_mse = min(accepted_val_mse, best_val_mse)
                break

            # Lock in the best for current depth as the latest accepted baseline
            self._restore(best_state)
            accepted_snapshot = best_state
            accepted_val_mse = best_val_mse

            # ───── Plateau reached → start a *series* of depth expansions with patience ─────
            pre_exp_snapshot = accepted_snapshot  # baseline before the series
            pre_exp_val_mse = accepted_val_mse
            pre_sizes = self._get_sizes()

            logger.info(
                f"Starting depth-expansion series from baseline val-MSE {pre_exp_val_mse:.6f}; "
                f"patience(trials_depth)={self.trials_depth}."
            )

            # Ensure we start expansions from the accepted baseline weights
            self._restore(pre_exp_snapshot)

            # Attempt multiple expansions until accept or patience exhausts
            while True:
                # Capacity guard
                depth_fc1 = getattr(self.fc1, "num_layers", 1)
                depth_pq = getattr(self.pq_fc2, "num_layers", 1)
                depth_vm = getattr(self.vm_fc2, "num_layers", 1)
                if depth_fc1 >= max_depth or depth_pq >= max_depth or depth_vm >= max_depth:
                    logger.info("Hit max_depth; stop expanding.")
                    stop_all = True
                    break

                # 1) Deepen all hidden stacks by +1 layer each
                self.fc1.append_depth(1, device=device)
                for br in self.branches:
                    getattr(self, f"{br}_fc2").append_depth(1, device=device)
                logger.info("Deepened: added 1 layer to fc1 and each branch-fc2.")

                # 2) (re)size heads to match fc2 inputs (width unchanged; safe no-op)
                self.head_pq = _resize_head(self.head_pq, self.pq_fc2.out_features).to(device)
                self.head_vm = _resize_head(self.head_vm, self.vm_fc2.out_features).to(device)

                # 3) New optimizer/scheduler for the changed parameter set
                opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)

                # 4) Train the deepened model with inner early-stopping
                deep_best_mse = float("inf")
                deep_best_state = self.snapshot()
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
                        f"[Task {self.current_task}] (deepened) epoch {self.global_epoch_count:4d}  "
                        f"val-MSE {val_loss:.6f}  patience {patience}"
                    )

                    if val_loss < deep_best_mse - delta:
                        deep_best_mse = val_loss
                        deep_best_state = self.snapshot()
                        patience = patience_orig
                    else:
                        patience -= 1

                # If max_epochs hit during deep training, pick better of deep vs baseline and exit
                if stop_all:
                    if deep_best_mse < pre_exp_val_mse - delta:
                        self._restore(deep_best_state)
                        accepted_snapshot = deep_best_state
                        accepted_val_mse = deep_best_mse
                    else:
                        # rollback to baseline of the series
                        self._shrink_to(pre_sizes)
                        self._restore(pre_exp_snapshot)
                        accepted_snapshot = pre_exp_snapshot
                        accepted_val_mse = pre_exp_val_mse
                    break

                # 5) Post-expansion decision with *patience on expansion*
                if deep_best_mse < pre_exp_val_mse - delta:
                    # ACCEPT: keep, reset counter, refresh baseline, leave series
                    logger.info(
                        "Depth expansion accepted; resetting failure counter. "
                        f"Improved val-MSE {pre_exp_val_mse:.6f} → {deep_best_mse:.6f} (delta={delta})."
                    )
                    self._restore(deep_best_state)
                    accepted_snapshot = deep_best_state
                    accepted_val_mse = deep_best_mse
                    self._depth_failures = 0
                    break  # leave expansion *series*; go train this arch again
                else:
                    # FAIL: increment counter; decide retry vs rollback
                    self._depth_failures += 1
                    if self._depth_failures >= self.trials_depth:
                        logger.info(
                            f"Depth expansions without improvement reached {self.trials_depth}; "
                            "rolling back and stopping."
                        )
                        # Roll back architecture and weights to the series baseline
                        self._shrink_to(pre_sizes)
                        self._restore(pre_exp_snapshot)
                        accepted_snapshot = pre_exp_snapshot
                        accepted_val_mse = pre_exp_val_mse
                        stop_all = True
                        break
                    else:
                        logger.info(
                            f"No val improvement after depth expansion (trial {self._depth_failures}/"
                            f"{self.trials_depth}); trying another expansion."
                        )
                        # Keep capacity; start the next attempt from the best weights of this failed attempt
                        self._restore(deep_best_state)
                        # Loop continues to append another layer
                        continue

            # End expansion series
            if stop_all:
                break

            # After an accepted expansion, continue outer loop to re-train to plateau at the new depth
            # and potentially start another series.
            # Optimizer/scheduler may keep state, but recreate to follow project conventions
            opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)
            # Continue while True to next cycle
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

        # physics metrics (same as before)
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
