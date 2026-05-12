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
from types import SimpleNamespace
from typing import Dict, Optional

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def _gate(h: torch.Tensor, ts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Element‑wise gating ⇢ zero‑out neurons **unavailable** for this task."""
    valid = (ts <= h.new_tensor(DEN_4Heads.current_task)).logical_and(mask)
    return h * valid.to(h.dtype)

def current_device(explicit: Optional[torch.device] = None) -> torch.device:
    """Return `explicit` if given, otherwise *cuda* (if available) else *cpu*."""
    return explicit or torch.device("cuda" if torch.cuda.is_available() else "cpu")

from Dyn_DNN4OPF.utils.constraint_losses import (
    mean_constraint_violation,
    power_balance_residuals
)
from collections import deque
import torch
import torch.nn as nn

# ── penalty variant needs case bounds / indices / ybus ───────────────────────
from Dyn_DNN4OPF.data.opf_loader import load_case_bounds

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
_DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def gate_activations(h: torch.Tensor, ts: torch.Tensor, mask: torch.Tensor, current_task: torch.Tensor) -> torch.Tensor:
    """
    Zero out any neuron whose mask is False or whose timestamp > current_task.
    All tensors remain on the same device as `h`.
    """
    # build boolean mask of shape (h.size(1),)
    device = h.device
    valid = (ts.to(device) <= current_task.to(device)) & mask.to(device)
    return h * valid.to(h.dtype)

class DEN_4Heads(nn.Module):
    """Two-hidden-layer MLP with dynamic expansion + splitting (GPU-only)."""

    def __init__(self, cfg: SimpleNamespace):
        super().__init__()
        # ── device setup ───────────────────────────────────────────
        d = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = d

        # ── branches definition (2-head: pq for [PG,QG], vm for [VA,VM]) ─────
        self.branches = ["pq", "vm"]

        # ── dimensions ─────────────────────────────────────────────
        self.in_dim, self.h1_dim, self.h2_dim = cfg.dims
        self.n_gen, self.n_bus = cfg.n_gen, cfg.n_bus

        # ── training hyperparams as CUDA buffers ───────────────────
        self.register_buffer("current_task", torch.tensor(0, dtype=torch.long,   device=d))
        self.register_buffer("loss_thr",     torch.tensor(cfg.loss_thr,  dtype=torch.float64, device=d))
        self.register_buffer("spl_thr",      torch.tensor(cfg.spl_thr,   dtype=torch.float64, device=d))
        self.register_buffer("gl_lambda",    torch.tensor(cfg.gl_lambda, dtype=torch.float64, device=d))
        self.register_buffer("l1_lambda",    torch.tensor(cfg.l1_lambda, dtype=torch.float64, device=d))

        # Penalty weights (scalars)
        self.lambda_loss = getattr(cfg, "lambda_loss", 1.0)
        self.lambda_eq   = getattr(cfg, "lambda_eq",   1.0)
        self.lambda_ineq = getattr(cfg, "lambda_ineq", 1.0)

        # ── static hyperparams (Python ints) ──────────────────────
        self.ex_k        = cfg.ex_k
        self.patience    = cfg.patience
        self.warm_epochs = cfg.warmup_epochs
        self.max_epochs  = cfg.max_epochs

        # ── backbone layers ───────────────────────────────────────
        self.fc1 = nn.Linear(self.in_dim, self.h1_dim, device=d)
        for br in self.branches:
            out_dim = (2 * self.n_gen) if br == "pq" else (2 * self.n_bus)
            setattr(self, f"{br}_fc2", nn.Linear(self.h1_dim, self.h2_dim, device=d))
            setattr(self, f"head_{br}", nn.Linear(self.h2_dim, out_dim,       device=d))

        # ── timestamp & mask buffers for gating ────────────────────
        self.register_buffer("fc1_mask",      torch.ones(self.h1_dim, dtype=torch.bool,  device=d))
        self.register_buffer("fc1_timestamp", torch.zeros(self.h1_dim, dtype=torch.long, device=d))
        for br in self.branches:
            self.register_buffer(f"{br}_fc2_mask",      torch.ones(self.h2_dim, dtype=torch.bool,  device=d))
            self.register_buffer(f"{br}_fc2_timestamp", torch.zeros(self.h2_dim, dtype=torch.long, device=d))

        # optional: stable IDs you can log against per head
        self.branch_task_id = {"pq": 1, "vm": 2}

        # ── drift snapshots (initialized on first split call) ────
        self._prev_W1: torch.Tensor | None = None
        self._prev_W2: Dict[str, torch.Tensor] = {}

        # ── physics constants on GPU (GPU‑first, single move via model.to(...)) ─
        raw = load_case_bounds(cfg.case_name)
        # counts (already in cfg, but keep authoritative from bounds)
        self.n_gen = int(raw["p_min"].numel())
        self.n_bus = int(raw["v_min"].numel())
        # bounds (float buffers)
        self.register_buffer("p_min", raw["p_min"].to(dtype=torch.float64))
        self.register_buffer("p_max", raw["p_max"].to(dtype=torch.float64))
        self.register_buffer("q_min", raw["q_min"].to(dtype=torch.float64))
        self.register_buffer("q_max", raw["q_max"].to(dtype=torch.float64))
        self.register_buffer("v_min", raw["v_min"].to(dtype=torch.float64))
        self.register_buffer("v_max", raw["v_max"].to(dtype=torch.float64))
        # indices (long buffers)
        self.register_buffer("gen_bus_idx", torch.as_tensor(raw["gen_buses"], dtype=torch.long))
        self.register_buffer("load_bus_idx", torch.as_tensor(raw["load_buses"], dtype=torch.long))
        # Y‑bus representation (buffer tensors; tuple shape kept as python int pair)
        yb = raw["y_bus"]
        if torch.is_tensor(yb):
            self.register_buffer("y_bus_sparse", yb)
            self._ybus_kind = "sparse"
        else:
            vals, rows, cols, shape = yb
            self.register_buffer("ybus_vals", vals)
            self.register_buffer("ybus_rows", rows)
            self.register_buffer("ybus_cols", cols)
            self._ybus_shape = shape
            self._ybus_kind  = "tuple"


    def _init_buffers(self) -> None:
        """
        Ensure that all timestamp & mask buffers are registered on CUDA.
        Safe to call in __init__ or before any dynamic ops.
        """
        device = self.device
        # shared fc1
        if not hasattr(self, "fc1_mask"):
            self.register_buffer(
                "fc1_mask",
                torch.ones(self.h1_dim, dtype=torch.bool, device=device),
            )
        if not hasattr(self, "fc1_timestamp"):
            self.register_buffer(
                "fc1_timestamp",
                torch.zeros(self.h1_dim, dtype=torch.long, device=device),
            )
        # per‐branch fc2
        for br in self.branches:
            mask_name = f"{br}_fc2_mask"
            ts_name   = f"{br}_fc2_timestamp"
            if not hasattr(self, mask_name):
                self.register_buffer(
                    mask_name,
                    torch.ones(self.h2_dim, dtype=torch.bool, device=device),
                )
            if not hasattr(self, ts_name):
                self.register_buffer(
                    ts_name,
                    torch.zeros(self.h2_dim, dtype=torch.long, device=device),
                )

    def _gate(self, h: torch.Tensor, ts: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        valid = (ts <= self.current_task).logical_and(mask)
        return h * valid.to(h.dtype)

    def _ybus(self):
        if getattr(self, "_ybus_kind", None) == "sparse":
            return self.y_bus_sparse
        return (self.ybus_vals, self.ybus_rows, self.ybus_cols, self._ybus_shape)

    def _split_pred(self, y: torch.Tensor):
        """Split concatenated outputs into (pg, qg, va, vm)."""
        ng, nb = self.n_gen, self.n_bus
        pq = y[:, : 2 * ng]
        vmv = y[:, 2 * ng : 2 * ng + 2 * nb]
        pg = pq[:, :ng]
        qg = pq[:, ng:]
        va = vmv[:, :nb]
        vm = vmv[:, nb:]
        return pg, qg, va, vm

    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        Forward pass with gating by mask & timestamp.
        Concatenates outputs in PG | QG | VA | VM order.
        If `task_id` is provided, we only use it to set the gating buffer.
        """
        # ensure input on GPU
        x = x.to(self.device, non_blocking=True)

        # If caller specifies a task/head index, sync the gating buffer for gating logic
        if task_id is not None and hasattr(self, "current_task"):
            self.current_task.data.fill_(int(task_id))

        # ─── shared layer ───────────────────────────────────────────────
        w1 = self.fc1.weight * self.fc1_mask.unsqueeze(1)
        h1 = F.linear(x, w1, self.fc1.bias)
        h1 = F.relu(h1)
        h1 = self._gate(h1, self.fc1_timestamp, self.fc1_mask)

        # ─── branch‐specific layers & heads ─────────────────────────────
        outs: list[torch.Tensor] = []
        for br in self.branches:
            fc2 = getattr(self, f"{br}_fc2")
            ts2 = getattr(self, f"{br}_fc2_timestamp")
            m2  = getattr(self, f"{br}_fc2_mask")

            w2 = fc2.weight * m2.unsqueeze(1)
            h2 = F.linear(h1, w2, fc2.bias)
            h2 = F.relu(h2)
            h2 = self._gate(h2, ts2, m2)

            head = getattr(self, f"head_{br}")
            outs.append(head(h2))

        return torch.cat(outs, dim=1)

    def loss_fn(self, x: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        """Composite loss: MSE + equality residuals + inequality penalties (GPU‑first)."""
        y_pred = self(x)
        mse = F.mse_loss(y_pred, y_true, reduction="mean")

        # PD/QD from inputs (flattened as [PD | QD])
        pd = x[:, : self.n_bus]
        qd = x[:, self.n_bus : 2 * self.n_bus]
        pg, qg, va, vm = self._split_pred(y_pred)

        res_P, res_Q = power_balance_residuals(
            pg, qg, pd, qd, vm, va,
            y_bus=self._ybus(),
            gen_bus_idx=self.gen_bus_idx,
            load_bus_idx=self.load_bus_idx,
            n_bus=self.n_bus,
        )
        eq_norm = (res_P.pow(2) + res_Q.pow(2)).mean()

        def _viol(v, lo, hi):
            return torch.clamp_min(v - hi, 0) + torch.clamp_min(lo - v, 0)

        viol_p = _viol(pg, self.p_min, self.p_max)
        viol_q = _viol(qg, self.q_min, self.q_max)
        viol_v = _viol(vm, self.v_min, self.v_max)
        ineq_norm = viol_p.pow(2).mean() + viol_q.pow(2).mean() + viol_v.pow(2).mean()

        return (
            self.lambda_loss * mse
            + self.lambda_eq   * eq_norm
            + self.lambda_ineq * ineq_norm
        )

    def freeze_head(self, task_id: int) -> None:
        """Freeze parameters of the selected head (and optionally its branch fc2)."""
        br = self.branches[int(task_id)]
        for p in getattr(self, f"head_{br}").parameters():
            p.requires_grad_(False)
        # Optionally also freeze the branch's fc2:
        for p in getattr(self, f"{br}_fc2").parameters():
            p.requires_grad_(False)

    def unfreeze_head(self, task_id: int) -> None:
        """Unfreeze parameters of the selected head (and optionally its branch fc2)."""
        br = self.branches[int(task_id)]
        for p in getattr(self, f"head_{br}").parameters():
            p.requires_grad_(True)
        # Optionally unfreeze the branch's fc2 as well.
        for p in getattr(self, f"{br}_fc2").parameters():
            p.requires_grad_(True)

    @torch.no_grad()
    def prune_new_neurons(self, thresh: float = 1e-3) -> None:
        """
        Deactivate any *newly added* neurons for the *current* task whose group-L2 < thresh.
        Operates in-place on the registered mask buffers.
        """
        # Shared layer
        g1 = self.fc1.weight.data.pow(2).sum(dim=1).sqrt()
        is_new1 = (self.fc1_timestamp == self.current_task)
        self.fc1_mask[is_new1] &= (g1[is_new1] >= thresh)

        # Per-branch layers
        for br in self.branches:
            fc2   = getattr(self, f"{br}_fc2")
            mask2 = getattr(self, f"{br}_fc2_mask")
            ts2   = getattr(self, f"{br}_fc2_timestamp")
            g2 = fc2.weight.data.pow(2).sum(dim=1).sqrt()
            is_new2 = (ts2 == self.current_task)
            mask2[is_new2] &= (g2[is_new2] >= thresh)

    def on_epoch_end(self) -> None:
        """
        Called once per epoch after optimizer.step().
        1) Split any drifting neurons.
        2) Prune any weak, newly‐added neurons.
        """
        self._split_drift()
        self.prune_new_neurons()

    @property
    def total_neurons(self) -> int:
        """Total count of neurons in shared + all branch‐specific fc2 layers."""
        count = self.fc1.weight.size(0)
        for br in self.branches:
            count += getattr(self, f"{br}_fc2").weight.size(0)
        return count

    @torch.no_grad()
    def _split_drift(self) -> None:
        """
        Detect semantic drift (L2 change > self.spl_thr) and duplicate+freeze
        both in shared fc1 and each branch fc2.
        """
        device = self.device

        # 1) Initialize snapshots if first call
        if self._prev_W1 is None:
            self._prev_W1 = self.fc1.weight.data.clone()
            for br in self.branches:
                self._prev_W2[br] = getattr(self, f"{br}_fc2").weight.data.clone()
            return

        # 2) Shared‐layer drift
        drift1 = (self.fc1.weight.data - self._prev_W1).pow(2).sum(1).sqrt()
        for idx in drift1.gt(self.spl_thr).nonzero(as_tuple=False).flatten().tolist():
            self._duplicate_neuron_fc1(idx)
        self._prev_W1.copy_(self.fc1.weight.data)

        # 3) Branch‐layer drift
        for br in self.branches:
            fc2 = getattr(self, f"{br}_fc2")
            prev = self._prev_W2[br]
            drift = (fc2.weight.data - prev).pow(2).sum(1).sqrt()
            for idx in drift.gt(self.spl_thr).nonzero(as_tuple=False).flatten().tolist():
                self._duplicate_neuron_branch(br, idx)
            self._prev_W2[br] = fc2.weight.data.clone()

    @torch.no_grad()
    def _duplicate_neuron_fc1(self, idx: int) -> None:
        """
        Clone & freeze shared‐layer neuron idx, timestamp new copy,
        and pad all branch‐fc2 inputs.
        """
        device = self.device
        # 1) Clone weights + bias
        W, b = self.fc1.weight.data, self.fc1.bias.data
        W_dup = torch.cat([W, W[idx:idx+1]], dim=0)
        b_dup = torch.cat([b, b[idx:idx+1]], dim=0)
        self.fc1 = nn.Linear(self.in_dim, W_dup.size(0), device=device)
        self.fc1.weight.data.copy_(W_dup)
        self.fc1.bias.data.copy_(b_dup)

        # 2) Freeze old, timestamp new
        self.fc1_mask[idx] = False
        self.fc1_mask = torch.cat([
            self.fc1_mask,
            torch.tensor([True], dtype=torch.bool, device=device)
        ], dim=0)
        self.fc1_timestamp = torch.cat([
            self.fc1_timestamp,
            torch.full((1,), self.current_task.item(), dtype=torch.long, device=device)
        ], dim=0)

        # 3) Pad every branch‐fc2 input
        for br in self.branches:
            fc2 = getattr(self, f"{br}_fc2")
            W2 = fc2.weight.data
            W2_pad = torch.zeros(W2.size(0), 1, device=device)
            W2_new = torch.cat([W2, W2_pad], dim=1)
            new_fc2 = nn.Linear(W2_new.size(1), W2_new.size(0), device=device)
            new_fc2.weight.data.copy_(W2_new)
            new_fc2.bias.data.copy_(fc2.bias.data)
            setattr(self, f"{br}_fc2", new_fc2)

    @torch.no_grad()
    def _duplicate_neuron_branch(self, br: str, idx: int) -> None:
        """
        Clone & freeze branch‐fc2 neuron idx, timestamp new copy,
        and pad only that branch’s head.
        """
        device = self.device
        # 1) Clone branch‐fc2
        fc2 = getattr(self, f"{br}_fc2")
        W, b = fc2.weight.data, fc2.bias.data
        W_dup = torch.cat([W, W[idx:idx+1]], dim=0)
        b_dup = torch.cat([b, b[idx:idx+1]], dim=0)
        new_fc2 = nn.Linear(W_dup.size(1), W_dup.size(0), device=device)
        new_fc2.weight.data.copy_(W_dup)
        new_fc2.bias.data.copy_(b_dup)
        setattr(self, f"{br}_fc2", new_fc2)

        # 2) Freeze old, timestamp new
        mask_name = f"{br}_fc2_mask"
        ts_name   = f"{br}_fc2_timestamp"
        old_mask = getattr(self, mask_name)
        old_ts   = getattr(self, ts_name)
        setattr(self, mask_name, torch.cat([
            old_mask,
            torch.tensor([True], dtype=torch.bool, device=device)
        ], dim=0))
        setattr(self, ts_name, torch.cat([
            old_ts,
            torch.full((1,), self.current_task.item(), dtype=torch.long, device=device)
        ], dim=0))
        getattr(self, mask_name)[idx] = False

        # 3) Pad branch head input
        head = getattr(self, f"head_{br}")
        Wh = head.weight.data
        Wh_pad = torch.zeros(Wh.size(0), 1, device=device)
        Wh_new = torch.cat([Wh, Wh_pad], dim=1)
        new_head = nn.Linear(Wh_new.size(1), Wh_new.size(0), device=device)
        new_head.weight.data.copy_(Wh_new)
        new_head.bias.data.copy_(head.bias.data)
        setattr(self, f"head_{br}", new_head)

    @torch.no_grad()
    def expand_layer(self,
                     layer: nn.Linear,
                     ex_k: Optional[int] = None
                     ) -> nn.Linear:
        """
        Add ex_k neurons to this Linear layer and correctly pad the next layer’s inputs,
        logging the before/after sizes.
        """
        logger = logging.getLogger(__name__)
        self.ex_k = ex_k if ex_k is not None else self.ex_k
        # original weights & bias
        W, b = layer.weight.data, layer.bias.data
        in_f, out_f = W.size(1), W.size(0)

        # decide which layer we're expanding for logging
        if layer is self.fc1:
            name = "fc1"
            layer_kind = "shared"
        else:
            layer_kind = None
            name = None
            br_hit = None
            for br in self.branches:
                if layer is getattr(self, f"{br}_fc2"):
                    name = f"{br}_fc2"
                    layer_kind = "branch"
                    br_hit = br
                    break
            if layer_kind is None:
                raise ValueError("expand_layer only supports fc1 or <branch>_fc2")

        # log before size
        logger.info(f"Expanding {name}: {out_f} → {out_f + self.ex_k} neurons")

        # 1) Make new rows for this layer
        new_W = torch.zeros(self.ex_k, in_f, device=W.device)
        new_b = torch.zeros(self.ex_k, device=b.device)

        # 2) Concatenate on the output dimension
        W2 = torch.cat([W, new_W], dim=0)
        b2 = torch.cat([b, new_b], dim=0)

        # 3) Rebuild this layer with larger output size
        new_layer = nn.Linear(in_f, out_f + self.ex_k, bias=True).to(W.device)
        new_layer.weight.data.copy_(W2)
        new_layer.bias.data.copy_(b2)

        # 4) Update timestamp & mask buffers
        if name == "fc1":
            old_ts = self.fc1_timestamp
            old_mask = self.fc1_mask
            new_ts = torch.cat([
                old_ts,
                torch.full((self.ex_k,), self.current_task.item(), dtype=torch.long, device=old_ts.device)
            ])
            new_mask = torch.cat([
                old_mask,
                torch.ones(self.ex_k, dtype=torch.bool, device=old_mask.device)
            ])
            # re-register buffers to keep them as buffers
            del self._buffers["fc1_timestamp"]
            del self._buffers["fc1_mask"]
            self.register_buffer("fc1_timestamp", new_ts)
            self.register_buffer("fc1_mask",      new_mask)
        else:
            # branch case
            old_ts = getattr(self, f"{br_hit}_fc2_timestamp")
            old_mask = getattr(self, f"{br_hit}_fc2_mask")
            new_ts = torch.cat([
                old_ts,
                torch.full((self.ex_k,), self.current_task.item(), dtype=torch.long, device=old_ts.device)
            ])
            new_mask = torch.cat([
                old_mask,
                torch.ones(self.ex_k, dtype=torch.bool, device=old_mask.device)
            ])
            del self._buffers[f"{br_hit}_fc2_timestamp"]
            del self._buffers[f"{br_hit}_fc2_mask"]
            self.register_buffer(f"{br_hit}_fc2_timestamp", new_ts)
            self.register_buffer(f"{br_hit}_fc2_mask",      new_mask)

        # 5) Grow the *next* layer’s input dimension
        if name == "fc1":
            # pad every branch fc2's input columns
            for br in self.branches:
                fc2 = getattr(self, f"{br}_fc2")
                Wc2 = fc2.weight.data
                pad2 = torch.zeros(Wc2.size(0), self.ex_k, device=Wc2.device)
                Wc2_new = torch.cat([Wc2, pad2], dim=1)
                new_fc2 = nn.Linear(Wc2_new.size(1), Wc2_new.size(0), bias=True).to(Wc2.device)
                new_fc2.weight.data.copy_(Wc2_new)
                new_fc2.bias.data.copy_(fc2.bias.data)
                setattr(self, f"{br}_fc2", new_fc2)
        else:
            # rebuild that branch's head so in_features becomes old_in + ex_k
            head = getattr(self, f"head_{br_hit}")
            old_in = head.in_features
            out_fh = head.out_features
            device_h = head.weight.device
            old_Wh = head.weight.data
            old_bh = head.bias.data

            new_head = nn.Linear(old_in + self.ex_k, out_fh).to(device_h)
            new_head.weight.data[:, :old_in] = old_Wh
            new_head.weight.data[:, old_in:] = 0
            new_head.bias.data.copy_(old_bh)

            setattr(self, f"head_{br_hit}", new_head)

        # swap layer reference on the module
        if name == "fc1":
            self.fc1 = new_layer
        else:
            setattr(self, f"{br_hit}_fc2", new_layer)

        return new_layer
    
    def snapshot_state(self) -> Dict[str, torch.Tensor]:
        """Return a full clone of all parameters & buffers."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def restore_state(self, snap: Dict[str, torch.Tensor]) -> None:
        """Load back a previously‐saved snapshot."""
        self.load_state_dict(snap)

    def on_task_end(self) -> None:
        """
        Hook to call at the end of each lifelong task:
        - split any remaining drifted neurons,
        - prune weak new neurons.
        """
        self._split_drift()
        self.prune_new_neurons()

    def train_one_epoch(
        self,
        loader,                         # iterable of (x, y, …)
        optim: torch.optim.Optimizer
    ) -> float:
        """
        One epoch over `loader` with masking/gating in forward().
        Returns average loss. All data & temporaries stay on self.device.
        """
        self.train()
        running = 0.0
        device = self.device

        for x, y, *_ in loader:
            # move batch to GPU once
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            loss = self.loss_fn(x, y)  # penalty loss needs x (PD/QD) + y
            loss.backward()
            optim.step()

            running += loss.item()

        return running / len(loader)


    def fit_task(
        self,
        train_loader,
        val_loader,
        *,
        max_epochs: int | None = None,
        delta: float | None = None
    ) -> float:
        """
        Lifelong‐task loop:
          1) warm‐up on MSE
          2) early‐stop w/ patience
          3) if best_val > loss_thr → expand & reset optim
          4) on_epoch_end() each epoch
        Returns best validation MSE.
        """
        device = self.device

        # ─── optimizer & scheduler setup ─────────────────
        if get_optimizer_scheduler is None:
            optim = torch.optim.Adam(self.parameters(), lr=getattr(self, "lr", 3e-4))
            sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)
        else:
            optim, sched = get_optimizer_scheduler(
                self.parameters(),
                lr=self.lr,
                **SCHEDULER_PARAMS
            )

        best_val     = float("inf")
        patience_left = self.patience
        delta         = float(delta) if delta is not None else float(self.loss_thr.item())

        # ─── 1) Warm‐up ────────────────────────────────────
        for _ in range(self.warm_epochs):
            self.train_one_epoch(train_loader, optim)
            self.on_epoch_end()

        # ─── 2) Early‐stop / expand loop ──────────────────
        epoch      = 0
        max_epochs = max_epochs or self.max_epochs

        while epoch < max_epochs and patience_left > 0:
            epoch += 1

            # train + step LR
            self.train_one_epoch(train_loader, optim)
            sched.step()
            self.on_epoch_end()

            # validation (pure MSE for comparability)
            self.eval()
            with torch.no_grad():
                val_loss = sum(
                    F.mse_loss(
                        self(x.to(device, non_blocking=True)),
                        y.to(device, non_blocking=True)
                    ).item()
                    for x, y, *_ in val_loader
                ) / len(val_loader)

            # early‐stop check
            if val_loss < best_val - delta:
                best_val    = val_loss
                best_state  = {k: v.clone() for k, v in self.state_dict().items()}
                patience_left = self.patience
            else:
                patience_left -= 1

            # expansion trigger?
            if best_val > self.loss_thr.item() and patience_left == 0:
                # rollback and expand
                self.load_state_dict(best_state)
                self.expand_layer(self.fc1)
                for br in self.branches:
                    self.expand_layer(getattr(self, f"{br}_fc2"))

                # new optimizer + scheduler
                if get_optimizer_scheduler is None:
                    optim = torch.optim.Adam(self.parameters(), lr=getattr(self, "lr", 3e-4))
                    sched = torch.optim.lr_scheduler.LambdaLR(optim, lambda _: 1.0)
                else:
                    optim, sched = get_optimizer_scheduler(
                        self.parameters(),
                        lr=self.lr,
                        **SCHEDULER_PARAMS
                    )

                patience_left = self.patience
                continue  # resume training after expansion

        # restore best weights before returning
        if "best_state" in locals():
            self.load_state_dict(best_state)

        return best_val
