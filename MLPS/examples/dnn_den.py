"""
===============================================================================
Dynamically Expandable Network (DEN) Implementation
===============================================================================

This module implements the Dynamically Expandable Network (DEN) architecture and
training logic for lifelong learning scenarios as proposed by Yoon et al. (ICLR 2018).

--------------------------------------------------------------------------------
Problem Addressed:
    Lifelong learning with sequential tasks leads to catastrophic forgetting and
    inefficient capacity usage. DEN addresses this by dynamically expanding the
    network capacity only when necessary, retraining selectively, and preventing
    semantic drift via neuron splitting and timestamping.

--------------------------------------------------------------------------------
Key Innovations:
    1. Selective Retraining:
        - Identify a minimal subnetwork relevant for the new task via sparse
          connectivity and BFS traversal.
        - Retrain only affected weights for efficiency and avoiding negative transfer.

    2. Dynamic Network Expansion:
        - Expand layers by adding neurons only when selective retraining fails
          to achieve a desired loss threshold.
        - Use group sparsity regularization to prune unnecessary units after expansion.

    3. Semantic Drift Prevention:
        - Measure drift for each neuron as L2 norm between old and new weights.
        - Duplicate neurons that drift beyond a threshold to preserve previous
          task functionality.
        - Timestamp neurons with task arrival order; restrict their use at inference
          time to prevent drift effects.

--------------------------------------------------------------------------------
Core Components:
    ▸ Model Architecture:
        - Extends base MLP with dynamic layers.
        - Supports methods: add_task_head, expand_layer, duplicate_neuron, snapshot.

    ▸ Training Pipeline:
        1. Selective retraining on identified subnetwork.
        2. Check if loss below threshold; else expand network.
        3. Apply group sparsity to prune neurons.
        4. Detect drifting neurons and duplicate them.
        5. Timestamp neurons for task-specific inference.

    ▸ Regularization:
        - Uses L1 on weights for sparsity.

--------------------------------------------------------------------------------
Workflow Integration:
    - Called primarily by `trainer.py` during sequential task training.
    - Uses helper functions from `training_helpers.py` for retraining and expansion.
    - Evaluated on OPF task batches for continual learning experiments.

References:
    Jaehong Yoon, Eunho Yang, Jeongtae Lee, and Sung Ju Hwang,
    "Lifelong Learning with Dynamically Expandable Networks," ICLR 2018.

"""

from __future__ import annotations
import logging
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def current_device(explicit: Optional[torch.device] = None) -> torch.device:
    """Return `explicit` if given, otherwise *cuda* (if available) else *cpu*."""
    return explicit or torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Dyn_DNN4OPF.utils.constraint_losses import (
    mean_constraint_violation,
    power_balance_residuals
)
from collections import deque

class DEN(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.config      = config
        self.max_epochs  = config.max_epochs

        # ── dynamic width + depth from the run-script tuple ────────────────────
        dims                 = config.dims                 # (in_dim, w, w, …, w)
        self.in_dim          = dims[0]
        self.hidden_dims     = list(dims[1:])              # len = depth
        self.depth           = len(self.hidden_dims)
        self.n_classes       = config.n_classes
        self.ex_k            = config.ex_k                 # neurons to add on expand

        # ── generic hyper-parameters (unchanged) ───────────────────────────────
        self.lr          = config.lr
        self.loss_thr    = config.loss_thr
        self.dp_thr      = config.dp_thr
        self.dq_thr      = config.dq_thr
        self.pg_thr      = config.pg_thr
        self.qg_thr      = config.qg_thr
        self.vm_thr      = config.vm_thr
        self.gap_thr     = config.gap_thr
        self.spl_thr     = config.spl_thr
        self.warmup_epochs = config.warmup_epochs
        self.patience      = config.patience

        # ── build hidden layers of equal width ────────────────────────────────
        self.layers = nn.ModuleList()
        in_f = self.in_dim
        for h in self.hidden_dims:
            self.layers.append(nn.Linear(in_f, h))
            in_f = h
        self.head = nn.Linear(in_f, self.n_classes)

        # ── one mask + timestamp buffer per hidden layer ───────────────────────
        self.masks      = []
        self.timestamps = []
        for idx, h in enumerate(self.hidden_dims):
            self.register_buffer(f"mask_{idx}", torch.ones(h, dtype=torch.bool))
            self.register_buffer(f"ts_{idx}",   torch.zeros(h, dtype=torch.long))
            self.masks.append(getattr(self, f"mask_{idx}"))
            self.timestamps.append(getattr(self, f"ts_{idx}"))

        # task bookkeeping
        self.current_task    = None
        self.prev_state      = None
        self.violation_queue = deque()

    def forward(self, x):
        assert self.current_task is not None, "set model.current_task before forward()"
        for idx, layer in enumerate(self.layers):
            active = (self.timestamps[idx] <= self.current_task) & self.masks[idx]
            w = layer.weight * active.unsqueeze(1)
            b = layer.bias   * active.to(layer.bias.dtype)
            x = F.relu(F.linear(x, w, b))
        return self.head(x)


    @torch.no_grad()
    def snapshot(self):
        self.prev_hidden_w = [layer.weight.data.clone().cpu() for layer in self.layers]

    @torch.no_grad()
    def expand_layer(self, idx: int, ex_k: Optional[int] = None):
        ex_k  = ex_k or self.ex_k
        layer = self.layers[idx]

        W, b  = layer.weight.data, layer.bias.data
        in_f  = W.size(1)
        out_f = W.size(0)
        logger.info(f"Expanding layer {idx}: {out_f} → {out_f + ex_k} neurons")

        # grow this layer -------------------------------------------------------
        new_W = torch.zeros(ex_k, in_f, device=W.device)
        new_b = torch.zeros(ex_k,        device=b.device)
        bigger = nn.Linear(in_f, out_f + ex_k, bias=True).to(W.device)
        bigger.weight.data.copy_(torch.cat([W, new_W], dim=0))
        bigger.bias.data.copy_(torch.cat([b, new_b], dim=0))
        self.layers[idx] = bigger

        # update mask & timestamp ----------------------------------------------
        old_mask = self.masks[idx]
        old_ts   = self.timestamps[idx]
        self.register_buffer(
            f"mask_{idx}",
            torch.cat([old_mask, torch.ones(ex_k, dtype=torch.bool, device=old_mask.device)])
        )
        self.register_buffer(
            f"ts_{idx}",
            torch.cat([old_ts, torch.full((ex_k,), self.current_task, dtype=torch.long,
                                        device=old_ts.device)])
        )
        self.masks[idx]      = getattr(self, f"mask_{idx}")
        self.timestamps[idx] = getattr(self, f"ts_{idx}")

        # pad the next layer (or head) -----------------------------------------
        if idx + 1 < self.depth:
            nxt = self.layers[idx + 1]
            pad = torch.zeros(nxt.out_features, ex_k, device=W.device)
            nxt.weight.data = torch.cat([nxt.weight.data, pad], dim=1)
        else:
            old_head = self.head
            new_head = nn.Linear(old_head.in_features + ex_k, old_head.out_features).to(W.device)
            new_head.weight.data[:, :old_head.in_features] = old_head.weight.data
            new_head.weight.data[:, old_head.in_features:] = 0
            new_head.bias.data.copy_(old_head.bias.data)
            self.head = new_head
        return bigger

   
    @torch.no_grad()
    def _split_drifting_neurons(self) -> None:
        if not hasattr(self, "prev_hidden_w"):
            return
        for idx, layer in enumerate(self.layers):
            prev_W = self.prev_hidden_w[idx] if idx < len(self.prev_hidden_w) else None
            if prev_W is None:
                continue
            W_now   = layer.weight.data.cpu()
            rows    = min(W_now.size(0), prev_W.size(0))
            cols    = min(W_now.size(1), prev_W.size(1))
            drift   = (W_now[:rows, :cols] - prev_W[:rows, :cols]).norm(dim=1)
            for n in (drift > self.spl_thr).nonzero(as_tuple=True)[0].tolist():
                self.duplicate_neuron(idx, n)

    def snapshot_state(self):
        """Deep‐copy entire model state for rollback."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def restore_state(self, state: dict):
        """
        Restore model parameters and buffers from a snapshot, but only for keys
        whose shapes still match the current model. Skip any that have changed size.
        """
        own_state = self.state_dict()
        for name, param in state.items():
            if name not in own_state:
                # this key no longer exists
                continue
            if own_state[name].shape == param.shape:
                # copy only when shapes agree
                own_state[name].copy_(param)
            else:
                logger.warning(
                    f"Skipping restore of '{name}': "
                    f"checkpoint shape {tuple(param.shape)} != model shape {tuple(own_state[name].shape)}"
                )
        # now write back into the module (strict=False ignores missing/unexpected)
        self.load_state_dict(own_state, strict=False)
    
    def train_task(
        self,
        train_loader,
        constraints,
        optimizer,
        *,
        max_epochs: int = 10000
    ) -> float:
        """
        Single epoch of pure-MSE training on the selected subnetwork.
        Increments the global epoch counter and enforces the cap.
        """
        global GLOBAL_EPOCH_COUNT, MAX_EPOCHS
        if 'GLOBAL_EPOCH_COUNT' not in globals():
            GLOBAL_EPOCH_COUNT = 0
        if 'MAX_EPOCHS' not in globals():
            MAX_EPOCHS = max_epochs

        if GLOBAL_EPOCH_COUNT >= MAX_EPOCHS:
            return float('inf')
        GLOBAL_EPOCH_COUNT += 1

        device = next(self.parameters()).device
        self.train()
        total = 0.0
        for xb, yb, _ in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            loss = F.mse_loss(self(xb), yb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total += loss.item()
        return total / len(train_loader)

    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader,
        constraints,
        *,
        max_epochs: int = 10000,
        delta: float    = None
    ) -> float:
        """
        1) Warm-up for `warmup_epochs` with pure MSE.
        2) Vanilla early stopping on validation MSE.
        3) Expand *every* hidden layer if best MSE > `loss_thr`, then repeat.
        4) Enforce a single global epoch counter & cap.
        """
        global GLOBAL_EPOCH_COUNT, MAX_EPOCHS
        if 'GLOBAL_EPOCH_COUNT' not in globals():
            GLOBAL_EPOCH_COUNT = 0
        if 'MAX_EPOCHS' not in globals():
            MAX_EPOCHS = max_epochs

        device = current_device()
        optimizer, scheduler = get_optimizer_scheduler(
            self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
        )

        best_val = float('inf')
        counter  = self.patience
        delta    = delta if delta is not None else self.loss_thr

        # — warm-up — -----------------------------------------------------------
        for _ in range(self.warmup_epochs):
            if GLOBAL_EPOCH_COUNT >= MAX_EPOCHS:
                return best_val
            GLOBAL_EPOCH_COUNT += 1
            _ = self.train_task(train_loader, constraints, optimizer, max_epochs=max_epochs)

        # — main loop — ---------------------------------------------------------
        while counter > 0:
            if GLOBAL_EPOCH_COUNT >= MAX_EPOCHS:
                break
            GLOBAL_EPOCH_COUNT += 1

            _ = self.train_task(train_loader, constraints, optimizer, max_epochs=max_epochs)
            scheduler.step()

            # validation --------------------------------------------------------
            self.eval()
            val_sum = 0.0
            with torch.no_grad():
                for xb, yb, _ in val_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    val_sum += F.mse_loss(self(xb), yb).item()
            val_mse = val_sum / len(val_loader)

            if val_mse < best_val - delta:
                best_val   = val_mse
                best_state = self.snapshot_state()
                counter    = self.patience
            else:
                counter -= 1

            # dynamic expansion -------------------------------------------------
            if best_val > self.loss_thr:
                prev_state = self.snapshot_state()
                for i in range(self.depth):            # << changed line
                    self.expand_layer(i, self.ex_k)    # << changed line
                optimizer, scheduler = get_optimizer_scheduler(
                    self.parameters(), lr=self.lr, **SCHEDULER_PARAMS
                )
                continue
            else:
                break

        # rollback & return -----------------------------------------------------
        self.restore_state(best_state)
        return best_val

    def duplicate_neuron(self, idx: int, neuron_idx: int):
        layer = self.layers[idx]
        W, b  = layer.weight.data, layer.bias.data
        in_f  = W.size(1)
        out_f = W.size(0)

        # clone row -------------------------------------------------------------
        bigger = nn.Linear(in_f, out_f + 1).to(W.device)
        bigger.weight.data.copy_(torch.cat([W, W[neuron_idx:neuron_idx+1]], dim=0))
        bigger.bias.data.copy_(torch.cat([b, b[neuron_idx:neuron_idx+1]], dim=0))
        self.layers[idx] = bigger

        # update mask & timestamp ----------------------------------------------
        old_mask = self.masks[idx]
        old_ts   = self.timestamps[idx]
        old_mask[neuron_idx] = False            # freeze original
        self.register_buffer(
            f"mask_{idx}",
            torch.cat([old_mask, torch.tensor([True],  dtype=torch.bool,
                                            device=old_mask.device)]))
        self.register_buffer(
            f"ts_{idx}",
            torch.cat([old_ts,   torch.tensor([self.current_task], dtype=torch.long,
                                            device=old_ts.device)]))
        self.masks[idx]      = getattr(self, f"mask_{idx}")
        self.timestamps[idx] = getattr(self, f"ts_{idx}")

        # pad next layer / head -------------------------------------------------
        if idx + 1 < self.depth:
            nxt = self.layers[idx + 1]
            col = nxt.weight.data[:, neuron_idx:neuron_idx+1].clone()
            nxt.weight.data = torch.cat([nxt.weight.data, col], dim=1)
        else:
            old_head = self.head
            new_head = nn.Linear(old_head.in_features + 1, old_head.out_features).to(W.device)
            new_head.weight.data[:, :old_head.in_features] = old_head.weight.data
            new_head.weight.data[:, -1:]                  = 0
            new_head.bias.data.copy_(old_head.bias.data)
            self.head = new_head