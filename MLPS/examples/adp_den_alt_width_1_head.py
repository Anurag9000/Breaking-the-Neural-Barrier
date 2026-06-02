from __future__ import annotations

import logging
from copy import deepcopy
from typing import Dict, Iterable, Optional, Tuple, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
PRECISION_FMT = ".15e"

# ADP REVIEW (BEFORE REFACTOR)
# - Heads: 1. Alternating width/depth with rollback on failed expansions; not forward-only.
# - Deviations: per-expansion rollback, mixed patience; not following updated ADP_algorithms.md.


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


class ADP_ALT_WIDTH_1Head(nn.Module):
    """Alternating ADP (width first) for a 1-head MLP, forward-only expansions."""

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
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
        super().__init__()
        self.in_dim, self.out_dim = int(in_dim), int(out_dim)
        self.width = int(width)
        self._depth = int(max(1, depth))
        self.ex_k_width = int(ex_k)
        self.ex_k_depth = 1
        self.max_neurons = int(max_neurons)
        self.max_width = int(max_width) if max_width is not None else int(max_neurons)
        self.max_depth = int(max_depth)
        self.delta_width = float(delta)
        self.delta_depth = float(delta)
        self.patience_width_exp = int(patience_width)
        self.patience_depth_exp = int(patience_depth)
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        layers = []
        in_f = self.in_dim
        for _ in range(self._depth):
            layers.append(nn.Linear(in_f, self.width))
            in_f = self.width
        self.hidden_layers = nn.ModuleList(layers)
        self.head = nn.Linear(self.width, self.out_dim)
        self.to(self.device)

    @property
    def depth(self) -> int:
        return len(self.hidden_layers)

    def snapshot_arch_and_state(self) -> Dict[str, object]:
        return {"width": self.width, "depth": self.depth, "state_dict": deepcopy(self.state_dict())}

    def restore_arch_and_state(self, snap: Dict[str, object]) -> None:
        self._rebuild(int(snap["width"]), int(snap["depth"]))
        self.load_state_dict(deepcopy(snap["state_dict"]), strict=False)

    def _rebuild(self, width: int, depth: int) -> None:
        width = int(width)
        depth = int(max(1, depth))
        layers = []
        in_f = self.in_dim
        for _ in range(depth):
            layers.append(nn.Linear(in_f, width))
            in_f = width
        self.hidden_layers = nn.ModuleList(layers)
        self.head = nn.Linear(width, self.out_dim).to(self.device)
        self.width = width
        self._depth = depth
        self.to(self.device)

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
        self.head = _resize_linear(self.head, self.head.out_features, new_w)
        self.width = new_w
        self.to(self.device)

    def expand_depth(self, inc: int) -> None:
        for _ in range(int(inc)):
            self.hidden_layers.append(nn.Linear(self.width, self.width).to(self.device))
        self._depth = self.depth

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.to(self.device, non_blocking=True)
        for layer in self.hidden_layers:
            h = F.relu(layer(h))
        return self.head(h)

    def train_with_early_stopping(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
        optimizer_ctor: Callable[[Iterable], torch.optim.Optimizer] = lambda p: torch.optim.Adam(p, lr=1e-3),
        es_patience: int = 20,
        max_epochs: int = 10000,
    ) -> Tuple[float, Dict[str, torch.Tensor], int]:
        opt = optimizer_ctor(self.parameters())
        dtype = next(self.parameters()).dtype
        best_val = float("inf")
        best_snap = self.snapshot_arch_and_state()
        best_epoch = 0
        patience = int(es_patience)
        epochs = 0

        while patience > 0 and epochs < max_epochs:
            epochs += 1
            self.train()
            for batch in train_loader:
                xb, yb = batch[0], batch[1]
                xb = xb.to(self.device, dtype=dtype, non_blocking=True)
                yb = yb.to(self.device, dtype=dtype, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self(xb), yb)
                loss.backward()
                opt.step()

            self.eval()
            tot, count = 0.0, 0
            with torch.no_grad():
                for batch in val_loader:
                    xb, yb = batch[0], batch[1]
                    xb = xb.to(self.device, dtype=dtype, non_blocking=True)
                    yb = yb.to(self.device, dtype=dtype, non_blocking=True)
                    tot += F.mse_loss(self(xb), yb, reduction="sum").item()
                    count += yb.numel()
            val = tot / max(1, count)
            if epochs % 100 == 0:
                logger.info(f"Epoch {epochs:03d} | Val MSE: {val:{PRECISION_FMT}}")
            if val < best_val - 1e-12:
                best_val = val
                best_snap = self.snapshot_arch_and_state()
                best_epoch = epochs
                patience = int(es_patience)
            else:
                patience -= 1
                if patience <= 0:
                    logger.info(
                        f"Early stopping triggered at epoch {epochs} | best_epoch={best_epoch} | best_val={best_val:{PRECISION_FMT}}"
                    )

        if epochs >= max_epochs:
            logger.info(f"Max epochs reached at epoch {epochs} | best_epoch={best_epoch} | best_val={best_val:{PRECISION_FMT}}")
        self.restore_arch_and_state(best_snap)
        return best_val, best_snap["state_dict"], epochs

    def fit(
        self,
        train_loader: Iterable,
        val_loader: Iterable,
        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
        optimizer_ctor: Callable[[Iterable], torch.optim.Optimizer] = lambda p: torch.optim.Adam(p, lr=1e-3),
        max_global_epochs: int = 500,
        es_patience: int = 20,
    ) -> Dict[str, float]:
        best_val, _, e = self.train_with_early_stopping(train_loader, val_loader, loss_fn, optimizer_ctor, es_patience, max_global_epochs)
        best_snap = self.snapshot_arch_and_state()
        best_width = self.width
        best_depth = self.depth

        depth_saturated = False
        width_saturated = False
        mode = "width"
        logger.info(
            f"Initial training complete | epochs={e} | best_val={best_val:{PRECISION_FMT}} | width={best_width} | depth={best_depth}"
        )

        while not (depth_saturated and width_saturated):
            improved = False
            if mode == "width":
                logger.info(f"Starting width phase from width={self.width}, depth={self.depth}")
                width_fail = 0
                while (
                    width_fail < self.patience_width_exp
                    and self.width < self.max_width
                    and self.width < self.max_neurons
                ):
                    if self.width + self.ex_k_width > self.max_width or self.width + self.ex_k_width > self.max_neurons:
                        break
                    logger.info(f"Expanding width {self.width} -> {self.width + self.ex_k_width}")
                    self.expand_width(self.ex_k_width)
                    val, _, e = self.train_with_early_stopping(train_loader, val_loader, loss_fn, optimizer_ctor, es_patience, max_global_epochs)
                    prev_best = best_val
                    if val < prev_best - self.delta_width:
                        best_val = val
                        best_snap = self.snapshot_arch_and_state()
                        best_width = self.width
                        best_depth = self.depth
                        improved = True
                        width_fail = 0
                        logger.info(
                            f"Accepted width expansion | epochs={e} | candidate_val={val:{PRECISION_FMT}} | prev_best={prev_best:{PRECISION_FMT}} | delta={self.delta_width:{PRECISION_FMT}} | new_best={best_val:{PRECISION_FMT}} | width={best_width} | depth={best_depth}"
                        )
                    else:
                        width_fail += 1
                        logger.info(
                            f"Width expansion not improved | epochs={e} | candidate_val={val:{PRECISION_FMT}} | incumbent_best={prev_best:{PRECISION_FMT}} | delta={self.delta_width:{PRECISION_FMT}} | failures={width_fail}/{self.patience_width_exp}"
                        )
                if not improved:
                    width_saturated = True
                    logger.info("Width phase saturated without improvement.")
                self.restore_arch_and_state(best_snap)
                mode = "depth"
            else:
                logger.info(f"Starting depth phase from width={self.width}, depth={self.depth}")
                depth_fail = 0
                while (
                    depth_fail < self.patience_depth_exp
                    and self.depth < self.max_depth
                ):
                    if self.depth + self.ex_k_depth > self.max_depth:
                        break
                    logger.info(f"Expanding depth {self.depth} -> {self.depth + self.ex_k_depth}")
                    self.expand_depth(self.ex_k_depth)
                    val, _, e = self.train_with_early_stopping(train_loader, val_loader, loss_fn, optimizer_ctor, es_patience, max_global_epochs)
                    prev_best = best_val
                    if val < prev_best - self.delta_depth:
                        best_val = val
                        best_snap = self.snapshot_arch_and_state()
                        best_depth = self.depth
                        best_width = self.width
                        improved = True
                        depth_fail = 0
                        logger.info(
                            f"Accepted depth expansion | epochs={e} | candidate_val={val:{PRECISION_FMT}} | prev_best={prev_best:{PRECISION_FMT}} | delta={self.delta_depth:{PRECISION_FMT}} | new_best={best_val:{PRECISION_FMT}} | width={best_width} | depth={best_depth}"
                        )
                    else:
                        depth_fail += 1
                        logger.info(
                            f"Depth expansion not improved | epochs={e} | candidate_val={val:{PRECISION_FMT}} | incumbent_best={prev_best:{PRECISION_FMT}} | delta={self.delta_depth:{PRECISION_FMT}} | failures={depth_fail}/{self.patience_depth_exp}"
                        )
                if not improved:
                    depth_saturated = True
                    logger.info("Depth phase saturated without improvement.")
                self.restore_arch_and_state(best_snap)
                mode = "width"

        self.restore_arch_and_state(best_snap)
        return {"val_mse": float(best_val), "best_width": float(best_width), "best_depth": float(best_depth)}


# ADP REVIEW (AFTER REFACTOR)
# - Heads: 1 (single head).
# - Implements ADP_ALT_WIDTH forward-only: width phase then depth phase; expansions keep marching until patience hits, no per-attempt rollback; restore global best between phases.
# - Uses train_with_early_stopping (patience_es), expand_width/expand_depth with width/depth failure counts and delta thresholds; head-aware resizing.
