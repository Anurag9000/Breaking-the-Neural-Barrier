import copy
import logging
import os

import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd

from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS
from Dyn_DNN4OPF.models.dnn_den import DEN  # base class for shared attrs (n_bus, n_gen, etc.)
from Dyn_DNN4OPF.models.dnn_den import (
    mean_constraint_violation,
    power_balance_residuals,
)
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ════════════════════════════════════════════════════════════════════════
# helpers
# ════════════════════════════════════════════════════════════════════════

def _cfg_get(cfg, key, default):
    """Dictionary-style get with attribute fallback."""
    try:
        return cfg.get(key, default)
    except Exception:
        return getattr(cfg, key, default)

# ════════════════════════════════════════════════════════════════════════
# basic resize helpers
# ════════════════════════════════════════════════════════════════════════

def _resize_linear(old: nn.Linear, new_out: int, new_in: int | None = None):
    """Create a *new* Linear(new_in, new_out) layer and copy the overlapping
    weights / biases from *old*.
    """
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
    """Only change the *input* dimension of the head layer, keeping output.
    """
    return _resize_linear(old, old.out_features, new_in)


# ════════════════════════════════════════════════════════════════════════
# width / depth expansion utilities (generic, operate in‑place)
# ════════════════════════════════════════════════════════════════════════

def expand_width(model: "ADPWidth", inc: int):
    """Increase *every* hidden layer by *inc* neurons (fan‑in of successors fixed)."""
    for i in range(len(model.hidden_layers)):
        cur = model.hidden_layers[i]
        # grow current layer's out_features
        new_out = cur.out_features + inc
        model.hidden_layers[i] = _resize_linear(cur, new_out)
        # fix next layer's fan‑in (head counts as last successor)
        if i + 1 < len(model.hidden_layers):
            nxt = model.hidden_layers[i + 1]
            model.hidden_layers[i + 1] = _resize_linear(nxt, nxt.out_features, new_out)
        else:  # last hidden‑>head
            model.head = _resize_head(model.head, new_out)


def expand_depth(model: "ADPWidth") -> None:
    """
    Insert **one new square hidden layer** (d×d) right before the head.
    Works whether the current last layer is rectangular (init_depth = 1)
    or square (init_depth ≥ 2).

    Post-condition:
        • hidden_layers grows by +1
        • head fan-in remains d
    """
    last = model.hidden_layers[-1]
    d    = last.out_features            # common width

    # only requirement now: head takes the same width d
    assert model.head.in_features == d, (
        "expand_depth expects the head's input size to match the "
        "output width of the last hidden layer."
    )

    device = last.weight.device
    model.hidden_layers.append(nn.Linear(d, d, bias=True).to(device))


def train_early_stop(
    model: "ADPWidth",
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
    best_model = copy.deepcopy(model)

    while model.global_epoch < max_epochs:
        # ── Train one epoch ────────────────────────────────────────────────
        model.global_epoch += 1
        model.train()
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.mse_loss(model(xb), yb)
            opt.zero_grad()
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
                xb, yb = xb.to(device), yb.to(device)
                tot += F.mse_loss(model(xb), yb).item()
        val = tot / len(val_loader)

        # ── Update early stopping state ───────────────────────────────────
        if val < best_val - delta:
            best_val   = val
            best_model = copy.deepcopy(model)
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
        
    # Restore best weights before returning
    model._restore(best_model)
    return best_val

# ════════════════════════════════════════════════════════════════════════
#  ADP-Width: adaptive width-first network with patience on width & depth
# ════════════════════════════════════════════════════════════════════════

class ADPWidth(DEN):
    """Adaptive network that expands **width first** (patience-controlled),
    and after each accepted width step, tries **depth** sweeps (also patience-controlled).
    Stops when width patience is exhausted or capacity guards trip.
    """

    # ─────────────────────────── init ───────────────────────────
    def __init__(self, config):
        super().__init__(config)              # ← DEN already built the net

        # expose the list built by DEN under the old name expected below
        self.hidden_layers = self.layers      # <-- ADDED alias

        # ---------- hyper-parameters & guards -------------------
        self.lr           = _cfg_get(config, 'lr', getattr(self, 'lr', 1e-3))
        self.delta        = _cfg_get(config, 'delta', 0.0)
        self.patience     = _cfg_get(config, 'patience', 20)
        self.ex_k         = _cfg_get(config, 'ex_k', 8)
        self.max_neurons  = _cfg_get(config, 'max_neurons', float('inf'))
        self.max_width    = _cfg_get(config, 'max_width', float('inf'))
        self.max_depth    = _cfg_get(config, 'max_depth', float('inf'))
        # patience knobs for expansion
        self.trials_width = _cfg_get(config, 'trials_width', 5)
        self.trials_depth = _cfg_get(config, 'trials_depth', 5)
        self.global_epoch = 0

        # failure counters
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
            
    # ───────────────────────── forward ─────────────────────────
    def forward(self, x):
        for layer in self.hidden_layers:
            x = F.relu(layer(x))
        x = self.head(x)
        return self.bound_layer(x)

    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        constraints=None,
        *,
        max_epochs: int = 10000,
        delta: float | None = None,
    ) -> float:
        # resolve delta (None → 0.0)
        dtmp = self.delta if delta is None else delta
        Δ    = 0.0 if dtmp is None else float(dtmp)
        P    = int(self.patience)
        inc  = int(self.ex_k)

        # initial baseline = current model's val loss (no training)
        device = next(self.parameters()).device
        self.eval()
        with torch.no_grad():
            tot = 0.0
            for xb, yb, _ in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                tot += F.mse_loss(self(xb), yb).item()
        best_val   = tot / len(val_loader)
        best_model = copy.deepcopy(self)

        # reset counters at task start
        self._width_failures = 0
        self._depth_failures = 0

        stop_outer = False

        # ── OUTER LOOP: width series (with patience) ─────────────────────
        while not stop_outer:
            # Baseline for this width series (MSE-based)
            pre_width_snapshot = copy.deepcopy(best_model)
            pre_width_val      = float(best_val)
            self._width_failures = 0

            # Keep attempting width expansions until accept/rollback/capacity
            while True:
                # capacity guard for width
                total_neurons = sum(l.out_features for l in self.hidden_layers)
                curr_width    = self.hidden_layers[-1].out_features
                if (
                    total_neurons + inc * len(self.hidden_layers) > self.max_neurons
                    or curr_width + inc > self.max_width
                ):
                    logger.info("Reached max_neurons / max_width — stop width expansion.")
                    stop_outer = True
                    break

                # try a width expansion
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
                    # accepted width → reset counter, update best and advance baseline
                    self._width_failures = 0
                    best_val   = w_val
                    best_model = copy.deepcopy(self)
                    pre_width_snapshot = copy.deepcopy(self)
                    pre_width_val      = best_val
                    logger.info("Width expansion accepted; resetting failure counter.")

                    # ── After each accepted width, run a DEPTH series with patience ──
                    pre_depth_snapshot = copy.deepcopy(self)
                    pre_depth_val      = best_val
                    self._depth_failures = 0

                    while True:
                        # depth capacity guard
                        if len(self.hidden_layers) >= self.max_depth:
                            logger.info("Reached max_depth — stop depth expansion at this width.")
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
                            # accept depth → reset counter and update baseline
                            self._depth_failures = 0
                            best_val   = d_val
                            best_model = copy.deepcopy(self)
                            pre_depth_snapshot = copy.deepcopy(self)
                            pre_depth_val      = best_val
                            logger.info("Depth expansion accepted at current width; counter reset.")
                            # try another depth increment (loop continues)
                            continue
                        else:
                            # failed depth attempt
                            self._depth_failures += 1
                            k = self._depth_failures
                            N = int(self.trials_depth)
                            if k < N:
                                logger.info(
                                    f"No val improvement after depth expansion (trial {k}/{N}); trying another depth expansion.")
                                # keep the added layer and immediately try another depth expansion
                                continue
                            else:
                                logger.info(
                                    f"Depth expansions without improvement reached {N}; rolling back depths and stopping depth series.")
                                # rollback to the baseline before this depth series
                                self._restore(pre_depth_snapshot)
                                break  # exit depth series, return to width loop

                    # after depth series ends, continue width series
                    continue

                else:
                    # failed width attempt
                    self._width_failures += 1
                    k = self._width_failures
                    N = int(self.trials_width)
                    if k < N:
                        logger.info(
                            f"No val improvement after width expansion (trial {k}/{N}); trying another width expansion.")
                        # keep the widened net and try another width increment
                        continue
                    else:
                        logger.info(
                            f"Width expansions without improvement reached {N}; rolling back widths and stopping.")
                        # rollback to baseline before this width series and stop training
                        self._restore(pre_width_snapshot)
                        stop_outer = True
                        break

        # ── final restore & update global counter ───────────────────────
        self._restore(best_model)
        ADPWidth.global_epoch = self.global_epoch  # persist for next task
        return best_val

    # ───────────────────── evaluation utils (unchanged) ─────────────────────
    def _evaluate_loader(self, loader, constraints, device):
        self.eval()
        tot_mse = 0.0
        with torch.no_grad():
            for Xb, Yb, *_ in loader:
                tot_mse += F.mse_loss(self(Xb.to(device)), Yb.to(device)).item()
        val_loss = tot_mse / len(loader)

        if constraints is None:
            return val_loss, {}

        # collect preds for constraint metrics
        X_all, Y_all = [], []
        with torch.no_grad():
            for Xb, *_ in loader:
                X_all.append(Xb.to(device))
                Y_all.append(self(Xb.to(device)))
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

    def _restore(self, snapshot):
        """
        Replace layers _and_ parameters with those in `snapshot`.
        """
        self.hidden_layers = torch.nn.ModuleList(
            [copy.deepcopy(l) for l in snapshot.hidden_layers]
        )
        self.head = copy.deepcopy(snapshot.head)
        # finally copy the weights/biases
        self.load_state_dict(snapshot.state_dict())
