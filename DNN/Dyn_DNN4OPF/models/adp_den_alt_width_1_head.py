"""
ADP-DEN Alternating (Width→Depth) — 1‑Head
ADP workflow, self‑contained. Width proposal first each cycle, then depth.
Acceptance: MSE‑first (improve by > delta). Tracks optional physics metric for reporting.
"""
from __future__ import annotations
import math
from copy import deepcopy
from typing import Callable, Dict, Iterable, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

# -------------------------------
# Helpers
# -------------------------------

def _copy_overlap_linear(src: nn.Linear, dst: nn.Linear) -> None:
    """Copy overlapping block src→dst (top‑left)."""
    with torch.no_grad():
        oh, ih = src.weight.shape; Oh, Ih = dst.weight.shape
        h = min(oh, Oh); w = min(ih, Ih)
        dst.weight[:h, :w].copy_(src.weight[:h, :w])
        if src.bias is not None and dst.bias is not None:
            dst.bias[:h].copy_(src.bias[:h])

class StackedLinear(nn.Module):
    """Depth‑expandable width×width stack with ReLU between internal layers."""
    def __init__(self, width: int, start_depth: int = 1):
        super().__init__()
        self.width = int(width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(max(1, start_depth))])
    @property
    def depth(self) -> int:
        return len(self.layers)
    def append_depth(self, n: int = 1) -> None:
        for _ in range(n):
            self.layers.append(nn.Linear(self.width, self.width))
    def shrink_to(self, d: int) -> None:
        while len(self.layers) > d:
            self.layers.pop()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

# -------------------------------
# Model (1‑Head)
# -------------------------------
class MLPAlt1Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 1):
        super().__init__()
        self.in_dim, self.out_dim = in_dim, out_dim
        self.width = int(width)
        self.fc1 = nn.Linear(in_dim, width)
        self.fc2 = StackedLinear(width, start_depth=max(1, depth))
        self.head = nn.Linear(width, out_dim)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x)) if self.fc2.depth > 1 else self.fc2(x)
        return self.head(x)
    # arch mgmt
    def arch_spec(self) -> Dict[str, int]:
        return {"width": self.width, "depth": self.fc2.depth}
    def snapshot(self) -> Tuple[Dict[str, int], Dict[str, torch.Tensor]]:
        return self.arch_spec(), deepcopy(self.state_dict())
    def _rebuild_width(self, neww: int) -> None:
        neww = int(neww)
        new_fc1 = nn.Linear(self.in_dim, neww); _copy_overlap_linear(self.fc1, new_fc1)
        new_fc2 = StackedLinear(neww, start_depth=self.fc2.depth)
        for i, old in enumerate(self.fc2.layers):
            _copy_overlap_linear(old, new_fc2.layers[i])
        new_head = nn.Linear(neww, self.out_dim); _copy_overlap_linear(self.head, new_head)
        self.fc1, self.fc2, self.head = new_fc1, new_fc2, new_head
        self.width = neww
    def restore(self, spec: Dict[str, int], state: Dict[str, torch.Tensor]) -> None:
        if spec["width"] != self.width:
            self._rebuild_width(spec["width"]) 
        if spec["depth"] != self.fc2.depth:
            d = spec["depth"]; cur = self.fc2.depth
            if d < cur: self.fc2.shrink_to(d)
            else: self.fc2.append_depth(d - cur)
        self.load_state_dict(state, strict=True)
    # expansions
    def widen_all(self, dk: int) -> None:
        self._rebuild_width(self.width + int(dk))
    def append_depth(self, n: int = 1) -> None:
        self.fc2.append_depth(n)

# -------------------------------
# Train / Eval
# -------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, val_loader: Iterable,
             loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             physics_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
             device: Optional[torch.device] = None) -> Tuple[float, Optional[float]]:
    model.eval(); tot, n = 0.0, 0; phys = 0.0
    for xb, yb in val_loader:
        xb = xb.to(device) if device else xb; yb = yb.to(device) if device else yb
        out = model(xb); loss = loss_fn(out, yb)
        tot += float(loss.detach()) * len(xb); n += len(xb)
        if physics_metric is not None:
            phys += float(physics_metric(out.detach().cpu(), yb.detach().cpu())) * len(xb)
    return tot / max(1, n), (phys / max(1, n)) if physics_metric is not None else None

def inner_train(model: nn.Module, train_loader: Iterable, val_loader: Iterable,
                loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                physics_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
                optimizer_ctor: Callable[[Iterable], torch.optim.Optimizer] = lambda p: torch.optim.Adam(p, lr=1e-3),
                es_patience: int = 20, device: Optional[torch.device] = None):
    opt = optimizer_ctor(model.parameters())
    best_v, best_p = math.inf, math.inf; best_state = deepcopy(model.state_dict())
    patience, epochs = es_patience, 0
    while True:
        epochs += 1
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device) if device else xb; yb = yb.to(device) if device else yb
            opt.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v, p = evaluate(model, val_loader, loss_fn, physics_metric, device)
        if v < best_v - 1e-12:
            best_v, best_p = v, p if p is not None else best_p
            best_state = deepcopy(model.state_dict()); patience = es_patience
        else:
            patience -= 1
            if patience <= 0: break
    model.load_state_dict(best_state)
    return best_v, best_p, best_state, epochs

# -------------------------------
# Controller (Alt‑Width)
# -------------------------------
class ADP_ALT_WIDTH_1Head:
    def __init__(self, in_dim: int, out_dim: int, width: int = 64, depth: int = 1,
                 ex_k: int = 8, max_neurons: int = 4096, max_depth: int = 16,
                 delta: float = 0.0, patience_width: int = 10, patience_depth: int = 10,
                 device: Optional[torch.device] = None):
        self.model = MLPAlt1Head(in_dim, out_dim, width, depth)
        self.device = device
        if device: self.model.to(device)
        self.ex_k, self.max_neurons, self.max_depth = int(ex_k), int(max_neurons), int(max_depth)
        self.delta = float(delta); self.pw, self.pd = int(patience_width), int(patience_depth)
        self._g_best = math.inf; self._g_phys = math.inf; self._g_snap = self.model.snapshot()

    def _upd(self, v: float, p: Optional[float]):
        if v < self._g_best - 1e-12:
            self._g_best = v; self._g_snap = self.model.snapshot()
        if p is not None and p < self._g_phys - 1e-12:
            self._g_phys = p; self._g_snap = self.model.snapshot()

    def fit(self, train_loader: Iterable, val_loader: Iterable,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
            physics_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
            optimizer_ctor: Callable[[Iterable], torch.optim.Optimizer] = lambda p: torch.optim.Adam(p, lr=1e-3),
            max_global_epochs: int = 500, es_patience: int = 20):
        tot = 0
        base_v, base_p, _, e = inner_train(self.model, train_loader, val_loader, loss_fn, physics_metric, optimizer_ctor, es_patience, self.device)
        tot += e; self._upd(base_v, base_p)
        fw = fd = 0
        while tot < max_global_epochs and not (fw >= self.pw and fd >= self.pd):
            # WIDTH first
            if self.model.width >= self.max_neurons:
                fw = self.pw
            else:
                pre = self.model.snapshot(); pv = base_v
                self.model.widen_all(self.ex_k)
                v, p, _, e = inner_train(self.model, train_loader, val_loader, loss_fn, physics_metric, optimizer_ctor, es_patience, self.device)
                tot += e
                if v < pv - self.delta:
                    base_v, base_p, fw = v, p, 0; self._upd(v, p)
                else:
                    self.model.restore(*pre); fw += 1
            # DEPTH next
            if self.model.fc2.depth >= self.max_depth:
                fd = self.pd
            else:
                pre = self.model.snapshot(); pv = base_v
                self.model.append_depth(1)
                v, p, _, e = inner_train(self.model, train_loader, val_loader, loss_fn, physics_metric, optimizer_ctor, es_patience, self.device)
                tot += e
                if v < pv - self.delta:
                    base_v, base_p, fd = v, p, 0; self._upd(v, p)
                else:
                    self.model.restore(*pre); fd += 1
        self.model.restore(*self._g_snap)
        return {"val_mse": float(self._g_best), "physics": float(self._g_phys)}
