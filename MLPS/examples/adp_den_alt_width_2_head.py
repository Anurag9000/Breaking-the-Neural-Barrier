from __future__ import annotations

import logging
import math
from copy import deepcopy
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ADP REVIEW (BEFORE REFACTOR)
# - Heads: 2 (PQ, VM), width-phase then depth-phase with per-attempt rollback.
# - Deviations: rollback on failed expansions, stop exploring after rollback, not forward-only per ADP_algorithms.md.


def _copy_overlap_linear(src: nn.Linear, dst: nn.Linear) -> None:
    with torch.no_grad():
        oh, ih = src.weight.shape
        Oh, Ih = dst.weight.shape
        h = min(oh, Oh)
        w = min(ih, Ih)
        dst.weight[:h, :w].copy_(src.weight[:h, :w])
        if src.bias is not None and dst.bias is not None:
            dst.bias[:h].copy_(src.bias[:h])


class StackedLinear(nn.Module):
    def __init__(self, width: int, start_depth: int = 1):
        super().__init__()
        self.width = int(width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(max(1, start_depth))])

    @property
    def depth(self) -> int:
        return len(self.layers)

    def append_depth(self, n: int = 1) -> None:
        device = self.layers[0].weight.device if len(self.layers) else torch.device("cpu")
        for _ in range(n):
            self.layers.append(nn.Linear(self.width, self.width).to(device))

    def shrink_to(self, d: int) -> None:
        while len(self.layers) > d:
            self.layers.pop(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x


class MLPAlt2Head(nn.Module):
    """Shared input, two independent hidden stacks (PQ, VM) and heads."""

    def __init__(self, in_dim: int, out_pq: int, out_vm: int, width: int = 64, depth: int = 1):
        super().__init__()
        self.in_dim, self.width = int(in_dim), int(width)
        self.fc1 = nn.Linear(self.in_dim, self.width)
        self.pq_fc2 = StackedLinear(self.width, start_depth=max(1, depth))
        self.vm_fc2 = StackedLinear(self.width, start_depth=max(1, depth))
        self.head_pq = nn.Linear(self.width, out_pq)
        self.head_vm = nn.Linear(self.width, out_vm)

    @property
    def depth(self) -> int:
        return self.pq_fc2.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        pq = self.pq_fc2(x)
        pq = F.relu(pq) if self.pq_fc2.depth > 1 else pq
        vm = self.vm_fc2(x)
        vm = F.relu(vm) if self.vm_fc2.depth > 1 else vm
        return torch.cat([self.head_pq(pq), self.head_vm(vm)], dim=-1)

    def snapshot(self) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
        return {"width": self.width, "depth": self.depth}, deepcopy(self.state_dict())

    def _rebuild_width(self, new_width: int) -> None:
        device = self.fc1.weight.device
        new_width = int(new_width)
        new_fc1 = nn.Linear(self.in_dim, new_width).to(device)
        _copy_overlap_linear(self.fc1, new_fc1)

        new_pq = StackedLinear(new_width, self.pq_fc2.depth).to(device)
        new_vm = StackedLinear(new_width, self.vm_fc2.depth).to(device)
        for i in range(self.pq_fc2.depth):
            _copy_overlap_linear(self.pq_fc2.layers[i], new_pq.layers[i])
            _copy_overlap_linear(self.vm_fc2.layers[i], new_vm.layers[i])

        new_hpq = nn.Linear(new_width, self.head_pq.out_features).to(device)
        _copy_overlap_linear(self.head_pq, new_hpq)
        new_hvm = nn.Linear(new_width, self.head_vm.out_features).to(device)
        _copy_overlap_linear(self.head_vm, new_hvm)

        self.fc1, self.pq_fc2, self.vm_fc2, self.head_pq, self.head_vm = new_fc1, new_pq, new_vm, new_hpq, new_hvm
        self.width = new_width
        self.to(device)

    def restore(self, spec: Dict[str, int], state: Dict[str, torch.Tensor]) -> None:
        if spec["width"] != self.width:
            self._rebuild_width(spec["width"])
        d = spec["depth"]
        cur = self.depth
        if d < cur:
            self.pq_fc2.shrink_to(d)
            self.vm_fc2.shrink_to(d)
        elif d > cur:
            self.pq_fc2.append_depth(d - cur)
            self.vm_fc2.append_depth(d - cur)
        self.load_state_dict(state, strict=True)

    def widen_all(self, inc: int) -> None:
        self._rebuild_width(self.width + int(inc))

    def append_depth(self, n: int = 1) -> None:
        self.pq_fc2.append_depth(n)
        self.vm_fc2.append_depth(n)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader: Iterable,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    physics_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    device: Optional[torch.device] = None,
) -> Tuple[float, Optional[float]]:
    model.eval()
    model_dtype = next(model.parameters()).dtype
    tot, n = 0.0, 0
    phys = 0.0
    phys_n = 0
    for batch in val_loader:
        xb, yb = batch[0], batch[1]
        xb = xb.to(device, dtype=model_dtype) if device else xb.to(dtype=model_dtype)
        yb = yb.to(device, dtype=model_dtype) if device else yb.to(dtype=model_dtype)
        out = model(xb)
        loss = loss_fn(out, yb)
        weight = yb.numel()
        tot += float(loss.detach()) * weight
        n += weight
        if physics_metric is not None:
            phys += float(physics_metric(out.detach().cpu(), yb.detach().cpu())) * len(xb)
            phys_n += len(xb)
    return tot / max(1, n), (phys / max(1, phys_n)) if physics_metric is not None else None


def train_with_early_stopping(
    model: nn.Module,
    train_loader: Iterable,
    val_loader: Iterable,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    physics_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
    optimizer_ctor: Callable[[Iterable], torch.optim.Optimizer] = lambda p: torch.optim.Adam(p, lr=1e-3),
    es_patience: int = 20,
    device: Optional[torch.device] = None,
    max_epochs: int = 10000,
) -> Tuple[float, Optional[float], Dict[str, torch.Tensor], int]:
    opt = optimizer_ctor(model.parameters())
    model_dtype = next(model.parameters()).dtype
    best_val, best_phys = math.inf, math.inf
    best_state = deepcopy(model.state_dict())
    best_epoch = 0
    patience, epochs = es_patience, 0
    while epochs < max_epochs:
        epochs += 1
        model.train()
        for batch in train_loader:
            xb, yb = batch[0], batch[1]
            xb = xb.to(device, dtype=model_dtype) if device else xb.to(dtype=model_dtype)
            yb = yb.to(device, dtype=model_dtype) if device else yb.to(dtype=model_dtype)
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
        val, phys = evaluate(model, val_loader, loss_fn, physics_metric, device)
        if epochs % 100 == 0:
            logger.info(f"Epoch {epochs:03d} | Val MSE: {val:.15e}" + (f" | Phys: {phys:.15e}" if phys is not None else ""))
        if val < best_val - 1e-12:
            best_val, best_phys = val, phys if phys is not None else best_phys
            best_state = deepcopy(model.state_dict())
            best_epoch = epochs
            patience = es_patience
        else:
            patience -= 1
            if patience <= 0:
                logger.info(
                    f"Early stopping triggered at epoch {epochs} | best_epoch={best_epoch} | best_val={best_val:.15e}"
                )
                break
    if epochs >= max_epochs:
        logger.info(f"Max epochs reached at epoch {epochs} | best_epoch={best_epoch} | best_val={best_val:.15e}")
    model.load_state_dict(best_state)
    return best_val, (best_phys if physics_metric is not None else None), best_state, epochs


class ADP_ALT_WIDTH_2Head:
    """ADP_ALT_WIDTH forward-only for 2-head DEN."""

    def __init__(
        self,
        in_dim: int,
        out_pq: int,
        out_vm: int,
        width: int = 64,
        depth: int = 1,
        ex_k: int = 1,
        max_width: Optional[int] = None,
        max_neurons: int = 4096,
        max_depth: int = 5,
        delta: float = 0.0,
        patience_width: int = 10,
        patience_depth: int = 10,
        device: Optional[torch.device] = None,
    ):
        self.model = MLPAlt2Head(in_dim, out_pq, out_vm, width, depth)
        self.device = device
        if device:
            self.model.to(device)
        self.ex_k_width = int(ex_k)
        self.ex_k_depth = 1
        self.max_neurons = int(max_neurons)
        self.max_width = int(max_width) if max_width is not None else int(max_neurons)
        self.max_depth = int(max_depth)
        self.delta_width = float(delta)
        self.delta_depth = float(delta)
        self.patience_width_exp = int(patience_width)
        self.patience_depth_exp = int(patience_depth)

    def fit(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
        physics_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
        optimizer_ctor: Callable[[Iterable], torch.optim.Optimizer] = lambda p: torch.optim.Adam(p, lr=1e-3),
        max_global_epochs: int = 500,
        es_patience: int = 20,
    ) -> Dict[str, Optional[float]]:
        best_val, best_phys, _, e = train_with_early_stopping(
            self.model, train_loader, val_loader, loss_fn, physics_metric, optimizer_ctor, es_patience, self.device, max_global_epochs
        )
        best_snap = self.model.snapshot()
        best_width = self.model.width
        best_depth = self.model.depth
        width_saturated = False
        depth_saturated = False
        mode = "width"
        logger.info(
            f"Initial training complete | epochs={e} | best_val={best_val:.15e} | width={best_width} | depth={best_depth}"
        )

        while not (width_saturated and depth_saturated):
            improved = False
            if mode == "width":
                logger.info(f"Starting width phase from width={self.model.width}, depth={self.model.depth}")
                width_fail = 0
                while (
                    width_fail < self.patience_width_exp
                    and self.model.width < self.max_width
                    and self.model.width < self.max_neurons
                ):
                    if self.model.width + self.ex_k_width > self.max_width or self.model.width + self.ex_k_width > self.max_neurons:
                        break
                    logger.info(f"Expanding width {self.model.width} -> {self.model.width + self.ex_k_width}")
                    self.model.widen_all(self.ex_k_width)
                    val, phys, _, e = train_with_early_stopping(
                        self.model, train_loader, val_loader, loss_fn, physics_metric, optimizer_ctor, es_patience, self.device, max_global_epochs
                    )
                    if val < best_val - self.delta_width:
                        best_val = val
                        best_phys = phys if physics_metric is not None else best_phys
                        best_snap = self.model.snapshot()
                        best_width = self.model.width
                        best_depth = self.model.depth
                        improved = True
                        width_fail = 0
                        logger.info(
                            f"Accepted width expansion | epochs={e} | best_val={best_val:.15e} | width={best_width} | depth={best_depth}"
                        )
                    else:
                        width_fail += 1
                        logger.info(
                            f"Width expansion not improved | epochs={e} | val={val:.15e} | failures={width_fail}/{self.patience_width_exp}"
                        )
                if not improved:
                    width_saturated = True
                    logger.info("Width phase saturated without improvement.")
                self.model.restore(*best_snap)
                mode = "depth"
            else:
                logger.info(f"Starting depth phase from width={self.model.width}, depth={self.model.depth}")
                depth_fail = 0
                while (
                    depth_fail < self.patience_depth_exp
                    and self.model.depth < self.max_depth
                ):
                    if self.model.depth + self.ex_k_depth > self.max_depth:
                        break
                    logger.info(f"Expanding depth {self.model.depth} -> {self.model.depth + self.ex_k_depth}")
                    self.model.append_depth(self.ex_k_depth)
                    val, phys, _, e = train_with_early_stopping(
                        self.model, train_loader, val_loader, loss_fn, physics_metric, optimizer_ctor, es_patience, self.device, max_global_epochs
                    )
                    if val < best_val - self.delta_depth:
                        best_val = val
                        best_phys = phys if physics_metric is not None else best_phys
                        best_snap = self.model.snapshot()
                        best_depth = self.model.depth
                        best_width = self.model.width
                        improved = True
                        depth_fail = 0
                        logger.info(
                            f"Accepted depth expansion | epochs={e} | best_val={best_val:.15e} | width={best_width} | depth={best_depth}"
                        )
                    else:
                        depth_fail += 1
                        logger.info(
                            f"Depth expansion not improved | epochs={e} | val={val:.15e} | failures={depth_fail}/{self.patience_depth_exp}"
                        )
                if not improved:
                    depth_saturated = True
                    logger.info("Depth phase saturated without improvement.")
                self.model.restore(*best_snap)
                mode = "width"

        self.model.restore(*best_snap)
        return {
            "val_mse": float(best_val),
            "physics": float(best_phys) if physics_metric is not None and best_phys is not None else None,
            "best_width": float(best_width),
            "best_depth": float(best_depth),
        }


# ADP REVIEW (AFTER REFACTOR)
# - Heads: 2 (PQ, VM).
# - Implements ADP_ALT_WIDTH forward-only: width-phase then depth-phase; failed expansions are kept, only global best restored between phases/end.
# - Acceptance: val loss improvements vs delta thresholds; tracks optional physics metric; uses train_with_early_stopping and head-aware expand_width/expand_depth.
