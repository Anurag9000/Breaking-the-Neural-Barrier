from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from Dyn_DNN4OPF.utils.optim_sched import get_optimizer_scheduler
from config import SCHEDULER_PARAMS

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ADP REVIEW (BEFORE REFACTOR)
# - Heads: 4 (Pg, Qg, Va, Vm via ADPBase_4Head/DEN_4head branches).
# - Mode: width-outer/depth-inner with per-attempt rollback; conflicts with forward-only spec.
# - Deviations vs updated ADP_algorithms.md: roll back on failed expansions instead of marching forward and restoring only best at end.


def _resize_linear(old: nn.Linear, new_out: int, new_in: Optional[int] = None) -> nn.Linear:
    if new_in is None:
        new_in = old.in_features
    new = nn.Linear(new_in, new_out, bias=True).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        new.bias[:r] = old.bias[:r]
    return new


def _resize_head(old: nn.Linear, new_in: int) -> nn.Linear:
    return _resize_linear(old, old.out_features, new_in)


class ADPWidth_4Head(nn.Module):
    """ADP_WIDTH_OUTER_DEPTH_INNER with forward-only expansions (Pg, Qg, Va, Vm)."""

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        dims = getattr(config, "dims", None)
        self.in_dim = getattr(config, "in_dim", None) or (dims[0] if dims else None)
        self.n_gen = int(getattr(config, "n_gen", getattr(config, "num_gens", 1)))
        self.n_bus = int(getattr(config, "n_bus", getattr(config, "num_buses", 1)))
        if self.in_dim is None:
            self.in_dim = int(2 * self.n_bus)
        if dims and len(dims) > 1 and (self.n_gen == 1 and self.n_bus == 1):
            out_dim = dims[-1]
            self.n_bus = max(1, out_dim // 4)
            self.n_gen = max(1, (out_dim - 2 * self.n_bus) // 2)
        default_width = dims[1] if dims and len(dims) > 1 else getattr(config, "width", 64)
        default_depth = len(dims) - 1 if dims and len(dims) > 2 else getattr(config, "depth", 1)
        self.width = int(default_width)
        self._depth = int(max(1, default_depth))

        self.lr = float(getattr(config, "lr", 1e-3))
        self.max_epochs = int(getattr(config, "max_epochs", 10_000))
        self.patience_es = int(getattr(config, "patience_es", getattr(config, "patience", 20)))
        self.patience_width_exp = int(getattr(config, "patience_width_exp", getattr(config, "trials_width", 5)))
        self.patience_depth_exp = int(getattr(config, "patience_depth_exp", getattr(config, "trials_depth", 5)))
        self.delta_width = float(getattr(config, "delta_width", getattr(config, "delta", 0.0) or 0.0))
        self.delta_depth = float(getattr(config, "delta_depth", getattr(config, "delta", 0.0) or 0.0))
        self.ex_k_width = int(getattr(config, "ex_k_width", getattr(config, "ex_k", 1)))
        self.ex_k_depth = int(getattr(config, "ex_k_depth", 1))
        self.max_width = int(getattr(config, "max_width", getattr(config, "max_neurons", 4096)))
        self.max_neurons = int(getattr(config, "max_neurons", self.max_width))
        self.max_depth = int(getattr(config, "max_depth", getattr(config, "max_layers", 32)))

        self._build_network(self.width, self._depth)

    @property
    def depth(self) -> int:
        return len(self.hidden_layers)

    def _build_network(self, width: int, depth: int) -> None:
        width = int(width)
        depth = int(max(1, depth))
        layers = []
        in_f = self.in_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_f, width))
            in_f = width
        self.hidden_layers = nn.ModuleList(layers)
        self.head_pg = nn.Linear(width, self.n_gen)
        self.head_qg = nn.Linear(width, self.n_gen)
        self.head_va = nn.Linear(width, self.n_bus)
        self.head_vm = nn.Linear(width, self.n_bus)
        self.width = width
        self._depth = depth
        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    def snapshot_arch_and_state(self) -> Dict[str, object]:
        return {"width": self.width, "depth": self.depth, "state_dict": deepcopy(self.state_dict())}

    def restore_arch_and_state(self, snap: Dict[str, object]) -> None:
        self._build_network(int(snap["width"]), int(snap["depth"]))
        self.load_state_dict(deepcopy(snap["state_dict"]), strict=False)

    def expand_width(self, inc: int) -> None:
        new_w = self.width + int(inc)
        if new_w > self.max_width or new_w > self.max_neurons:
            return
        new_layers = []
        in_f = self.in_dim
        for layer in self.hidden_layers:
            new_layers.append(_resize_linear(layer, new_w, in_f))
            in_f = new_w
        self.hidden_layers = nn.ModuleList(new_layers)
        self.head_pg = _resize_head(self.head_pg, new_w)
        self.head_qg = _resize_head(self.head_qg, new_w)
        self.head_va = _resize_head(self.head_va, new_w)
        self.head_vm = _resize_head(self.head_vm, new_w)
        self.width = new_w
        self.to(next(self.parameters()).device)

    def expand_depth(self, inc: int) -> None:
        for _ in range(int(inc)):
            self.hidden_layers.append(nn.Linear(self.width, self.width).to(next(self.parameters()).device))
        self._depth = self.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.to(next(self.parameters()).device, non_blocking=True)
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        return torch.cat([self.head_pg(h), self.head_qg(h), self.head_va(h), self.head_vm(h)], dim=-1)

    def train_with_early_stopping(
        self, train_loader: Iterable, val_loader: Iterable
    ) -> Tuple[float, Dict[str, torch.Tensor]]:
        opt, sched = get_optimizer_scheduler(self.parameters(), lr=self.lr, **SCHEDULER_PARAMS)
        best_val = float("inf")
        best_snap = self.snapshot_arch_and_state()
        patience = int(self.patience_es)
        epochs = 0
        device = next(self.parameters()).device
        dtype = next(self.parameters()).dtype

        while patience > 0 and epochs < self.max_epochs:
            epochs += 1
            self.train()
            for xb, yb, *rest in train_loader:
                xb = xb.to(device, dtype=dtype, non_blocking=True)
                yb = yb.to(device, dtype=dtype, non_blocking=True)
                loss = F.mse_loss(self(xb), yb)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()
            try:
                sched.step()
            except Exception:
                pass

            self.eval()
            tot = 0.0
            count = 0
            with torch.no_grad():
                for xb, yb, *rest in val_loader:
                    xb = xb.to(device, dtype=dtype, non_blocking=True)
                    yb = yb.to(device, dtype=dtype, non_blocking=True)
                    tot += F.mse_loss(self(xb), yb, reduction="sum").item()
                    count += yb.numel()
            val = tot / max(1, count)
            if epochs % 100 == 0:
                logger.info("ES epoch %d val=%.15e patience=%d", epochs, val, patience)
            if val < best_val - 1e-12:
                best_val = val
                best_snap = self.snapshot_arch_and_state()
                patience = int(self.patience_es)
            else:
                patience -= 1

        self.restore_arch_and_state(best_snap)
        return best_val, best_snap["state_dict"]

    def _optimize_depth_at_fixed_width(
        self, train_loader: Iterable, val_loader: Iterable
    ) -> Tuple[float, Dict[str, torch.Tensor], int]:
        best_val, _ = self.train_with_early_stopping(train_loader, val_loader)
        best_snap = self.snapshot_arch_and_state()
        best_depth = self.depth
        depth_fail = 0
        while depth_fail < self.patience_depth_exp and self.depth < self.max_depth:
            if self.depth + self.ex_k_depth > self.max_depth:
                break
            self.expand_depth(self.ex_k_depth)
            val, _ = self.train_with_early_stopping(train_loader, val_loader)
            if val < best_val - self.delta_depth:
                best_val = val
                best_snap = self.snapshot_arch_and_state()
                best_depth = self.depth
                depth_fail = 0
            else:
                depth_fail += 1
                # keep deeper arch; no rollback
        self.restore_arch_and_state(best_snap)
        return best_val, best_snap["state_dict"], best_depth

    def fit_task(
        self,
        train_loader,
        val_loader,
        test_loader=None,
        constraints=None,
        *,
        max_epochs: Optional[int] = None,
    ) -> float:
        if max_epochs is not None:
            self.max_epochs = int(max_epochs)
        base_val, _ = self.train_with_early_stopping(train_loader, val_loader)
        best_val = base_val
        best_snap = self.snapshot_arch_and_state()
        best_width = self.width
        best_depth = self.depth
        width_fail = 0

        while (
            width_fail < self.patience_width_exp
            and self.width < self.max_width
            and self.width < self.max_neurons
        ):
            if self.width + self.ex_k_width > self.max_width or self.width + self.ex_k_width > self.max_neurons:
                break
            self.expand_width(self.ex_k_width)
            val_w, _, depth_w = self._optimize_depth_at_fixed_width(train_loader, val_loader)
            if val_w < best_val - self.delta_width:
                best_val = val_w
                best_snap = self.snapshot_arch_and_state()
                best_width = self.width
                best_depth = depth_w
                width_fail = 0
            else:
                width_fail += 1
                # keep widened arch; no rollback

        self.restore_arch_and_state(best_snap)
        self._best_width = best_width
        self._best_depth = best_depth
        return best_val


# ADP REVIEW (AFTER REFACTOR)
# - Heads: 4 (Pg, Qg, Va, Vm).
# - Implements ADP_WIDTH_OUTER_DEPTH_INNER with forward-only expansions (no per-attempt rollback); outer width expansions and inner depth search use failure counts vs patiences and delta thresholds.
# - Uses train_with_early_stopping (patience_es) and snapshot_arch_and_state / restore_arch_and_state; expand_width/expand_depth adjust hidden stack and all heads.
