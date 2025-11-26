import copy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from models.adp_base_4head import ADPBase_4Head
from Dyn_DNN4OPF.models.dnn_den_4head import (
    mean_constraint_violation,
    power_balance_residuals,
)
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ════════════════════════════════════════════════════════════════════════
# basic resize helpers
# ════════════════════════════════════════════════════════════════════════

def _cfg_get(cfg, key, default):
    """Dictionary-style access with attribute fallback."""
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)


def _resize_linear(old: nn.Linear, new_out: int, new_in: int | None = None):
    """Create a *new* Linear(new_in, new_out) layer and copy overlapping weights/biases."""
    if new_in is None:
        new_in = old.in_features
    new_layer = nn.Linear(new_in, new_out, bias=True).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new_layer.weight[:r, :c] = old.weight[:r, :c]
        new_layer.bias[:r] = old.bias[:r]
    return new_layer


def _resize_head(old: nn.Linear, new_in: int):
    """Only change the input dimension of the head layer, keeping output."""
    return _resize_linear(old, old.out_features, new_in)


def expand_width(model: "ADPWidth_4Head", inc: int) -> None:
    """
    Widen the network by `inc` neurons, uniformly:

        • every shared hidden layer in `model.hidden_layers`
        • every branch-specific second layer  pg_fc2 | qg_fc2 | va_fc2 | vm_fc2

    Padding of downstream inputs, masks, timestamps, and heads
    is handled by `model.expand_layer()` from ADPBase.
    """
    # ── Shared stack (depth ≥ 1) ─────────────────────────────────────────
    for lyr in model.hidden_layers:
        model.expand_layer(lyr, inc)

    # ── Branch-specific fc2 layers ───────────────────────────────────────
    for br in model.branches:
        model.expand_layer(getattr(model, f"{br}_fc2"), inc)


def expand_depth(model: "ADPWidth_4Head") -> None:
    """
    Append **one** new shared hidden layer (square: d × d) at the end of the
    shared stack.  Branch-specific layers are left unchanged.
    """
    width  = model.hidden_layers[-1].out_features
    device = model.device

    new_layer = nn.Linear(width, width, bias=True, device=device)
    model.hidden_layers.append(new_layer)

    # keep ADPBase’s legacy flat layer list up-to-date
    model._rebuild_layers_list()


def train_early_stop(
    model: "ADPWidth_4Head",
    train_loader,
    val_loader,
    *,
    patience: int,
    delta: float,
    max_epochs: int,
):
    """
    Train `model` on `train_loader`, early-stopping on validation MSE.
    *   `max_epochs` is a **global budget** across all calls; the function
        exits once `model.global_epoch` reaches that number.
    *   Prints one line per epoch:
          Epoch k | val_loss=… | width=… | depth=…
    Returns the best validation loss achieved.
    """
    device = next(model.parameters()).device
    opt, sch = get_optimizer_scheduler(
        model.parameters(), lr=model.lr, **SCHEDULER_PARAMS
    )

    best_val   = float("inf")
    counter    = patience
    best_model  = copy.deepcopy(model)

    while model.global_epoch < max_epochs:
        # ── Train one epoch ────────────────────────────────────────────────
        model.global_epoch += 1
        model.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        try:
            sch.step()
        except Exception:
            pass

        # ── Validate ───────────────────────────────────────────────────────
        model.eval()
        tot = 0.0
        with torch.no_grad():
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                tot += F.mse_loss(model(xb), yb).item()
        val = tot / len(val_loader)

        # ── Early-stopping bookkeeping ─────────────────────────────────────
        if val < best_val - 1e-12:  # strict improvement within phase
            best_val   = val
            best_model  = copy.deepcopy(model)
            counter    = patience            # reset
        else:
            counter -= 1                     # countdown

        # ── Per-epoch console logging (after update) ----------------------
        curr_width = model.hidden_layers[-1].out_features
        curr_depth = len(model.hidden_layers)
        logger.info(
            f"Epoch {model.global_epoch:5d} | "
            f"val_loss={val:.6f} | width={curr_width} | depth={curr_depth} | "
            f"patience_left={counter}"
        )

        if counter == 0:
            break

    # Restore phase-best weights before returning
    model._restore(best_model)
    return best_val

# ════════════════════════════════════════════════════════════════════════
#  ADP-Width: adaptive depth-&-width expanding network (+ expansion patience)
# ════════════════════════════════════════════════════════════════════════

class ADPWidth_4Head(ADPBase_4Head):
    """Adaptive network that first expands *width* until plateau, then tries
    depth sweeps at the current width. Adds **patience on expansion** for both
    width and depth.
    """

    # ─────────────────────────── init ───────────────────────────
    def __init__(self, config):
        super().__init__(config)

        # expose the list built by DEN under the old name expected below
        self.hidden_layers = self.layers      # alias

        # ---------- hyper-parameters & guards -------------------
        self.lr          = config.lr
        self.delta       = config.delta
        self.patience    = config.patience
        self.ex_k        = config.ex_k
        self.max_neurons = config.max_neurons
        self.max_width   = getattr(config, "max_width", float("inf"))
        self.max_depth   = getattr(config, "max_depth", float("inf"))
        self.global_epoch = 0

        # new: patience knobs (dict-style with defaults)
        self.trials_width = int(_cfg_get(config, "trials_width", 5))
        self.trials_depth = int(_cfg_get(config, "trials_depth", 5))
        self._width_failures = 0
        self._depth_failures = 0

        # ---------- optional bounded activation -----------------
        if all(hasattr(config, k) for k in ("bounds_low", "bounds_high", "mask")):
            self.bound_layer = BoundedAct(
                config.bounds_low, config.bounds_high, config.mask
            )
            self.bound_layer.apply_bounds.fill_(False)
        else:
            self.bound_layer = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pipeline:
            shared stack (depth ≥ 1) → branch-specific fc2 → heads → BoundedAct
        All tensors stay on GPU; mask / timestamp gating is applied in branch layers.
        """
        x = x.to(self.device, non_blocking=True)

        # ── shared stack ─────────────────────────────────────────────────────
        h = x
        for layer in self.hidden_layers:
            h = F.relu(layer(h))

        # ── branch-specific path ─────────────────────────────────────────────
        outs = []
        for br in self.branches:
            fc2 = getattr(self, f"{br}_fc2")
            ts2 = getattr(self, f"{br}_fc2_timestamp")
            m2  = getattr(self, f"{br}_fc2_mask")

            h2 = F.relu(F.linear(h, fc2.weight * m2.unsqueeze(1), fc2.bias))
            h2 = self._gate(h2, ts2, m2)
            outs.append(getattr(self, f"head_{br}")(h2))

        return self.bound_layer(torch.cat(outs, dim=1))

    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        constraints=None,
        *,
        max_epochs: int = 10_000,
        delta: float | None = None,
    ) -> float:
        """
        OUTER loop   → progressive **width** sweeps (shared + branch layers) with patience.
        INNER loop   → at each accepted width level, try adding **one shared layer** depth
                        with patience.
        Expansion is accepted IFF the post-phase **validation MSE** improves vs the
        *pre-expansion baseline* by at least `delta` (treat `None` as `0.0`).
        """
        dtmp   = self.delta if delta is None else delta
        Δ      = 0.0 if dtmp is None else float(dtmp)
        P      = int(self.patience)
        inc    = int(self.ex_k)
        best_V = float("inf")
        best_S = self.snapshot()  # deep copy via ADPBase

        # reset counters
        self._width_failures = 0
        self._depth_failures = 0

        stop_outer = False

        # ─────────────── OUTER LOOP  (width series with patience) ───────────────
        while not stop_outer:
            # Baseline for this width series: last accepted
            pre_width_snapshot = best_S
            pre_width_val      = best_V
            self._width_failures = 0

            while True:
                # capacity guards (shared + branches)
                total_neurons = (
                    sum(l.out_features for l in self.hidden_layers) +
                    sum(getattr(self, f"{br}_fc2").out_features for br in self.branches)
                )
                curr_width = self.hidden_layers[-1].out_features

                if (
                    total_neurons + inc * (len(self.hidden_layers) + len(self.branches)) > self.max_neurons
                    or curr_width + inc > self.max_width
                ):
                    logger.info("Reached max_neurons / max_width — stop widening.")
                    stop_outer = True
                    break

                # attempt a width increment
                expand_width(self, inc)
                w_val = train_early_stop(
                    self,
                    train_loader,
                    val_loader,
                    patience=P,
                    delta=Δ,
                    max_epochs=max_epochs,
                )

                if w_val < pre_width_val - Δ:
                    # ACCEPT width → reset counter, update best & advance baseline
                    self._width_failures = 0
                    best_V = w_val
                    best_S = self.snapshot()
                    pre_width_snapshot = best_S
                    pre_width_val      = best_V
                    logger.info("Width expansion accepted; resetting failure counter.")

                    # ─────────── INNER LOOP  (depth series with patience) ───────────
                    pre_depth_snapshot = best_S
                    pre_depth_val      = best_V
                    self._depth_failures = 0

                    while True:
                        if len(self.hidden_layers) >= self.max_depth:
                            logger.info("Reached max_depth — no further depth growth.")
                            break

                        expand_depth(self)
                        d_val = train_early_stop(
                            self,
                            train_loader,
                            val_loader,
                            patience=P,
                            delta=Δ,
                            max_epochs=max_epochs,
                        )

                        if d_val < pre_depth_val - Δ:
                            # ACCEPT depth → reset counter, update best & advance baseline
                            self._depth_failures = 0
                            best_V = d_val
                            best_S = self.snapshot()
                            pre_depth_snapshot = best_S
                            pre_depth_val      = best_V
                            logger.info("Depth expansion accepted; resetting failure counter.")
                            # try another depth step
                            continue
                        else:
                            # FAIL depth
                            self._depth_failures += 1
                            k, N = self._depth_failures, int(self.trials_depth)
                            if k < N:
                                logger.info(
                                    f"No val improvement after depth expansion (trial {k}/{N}); trying another depth expansion.")
                                # keep the added layer; sometimes stacked steps unlock gains
                                continue
                            else:
                                logger.info(
                                    f"Depth expansions without improvement reached {N}; rolling back depths and stopping depth series.")
                                self._restore(pre_depth_snapshot)
                                break

                    # after depth series, continue width series
                    continue

                else:
                    # FAIL width
                    self._width_failures += 1
                    k, N = self._width_failures, int(self.trials_width)
                    if k < N:
                        logger.info(
                            f"No val improvement after width expansion (trial {k}/{N}); trying another width expansion.")
                        # keep widened net; attempt another width increment
                        continue
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back widths and stopping.")
                        self._restore(pre_width_snapshot)
                        stop_outer = True
                        break

        # ─── end: restore best & propagate global epoch ───
        self._restore(best_S)
        ADPWidth_4Head.global_epoch = self.global_epoch
        return best_V


    # ───────────────────── evaluation utils (unchanged) ─────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()
        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device, non_blocking=True)), Yb.to(device, non_blocking=True)).item()
        val_loss = tot_mse / len(loader)

        if constraints is None:
            return val_loss, {}

        # collect preds for constraint metrics
        X_all, Y_all = [], []
        with torch.no_grad():
            for Xb, *_ in loader:
                X_all.append(Xb.to(device, non_blocking=True))
                Y_all.append(self(Xb.to(device, non_blocking=True)))
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

    # snapshot helpers for external caller (CSV logging etc.)
    def _csv_append(self, row, log_path):
        os.makedirs(os.path.dirname(log_path), exist_ok=True)
        df = pd.DataFrame([row])
        df.to_csv(log_path, mode="a", index=False, header=not os.path.exists(log_path))
