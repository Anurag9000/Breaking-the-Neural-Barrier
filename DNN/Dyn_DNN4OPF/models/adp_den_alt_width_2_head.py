"""
ADP-DEN Alternating (Width→Depth) — 2‑Head
Shared fc1 → branch stacks (PQ, VM) → concat heads. Width proposal first.
Acceptance: MSE‑first (improve by > delta). Physics metric optional.
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
    with torch.no_grad():
        oh, ih = src.weight.shape; Oh, Ih = dst.weight.shape
        h = min(oh, Oh); w = min(ih, Ih)
        dst.weight[:h, :w].copy_(src.weight[:h, :w])
        if src.bias is not None and dst.bias is not None:
            dst.bias[:h].copy_(src.bias[:h])

class StackedLinear(nn.Module):
    def __init__(self, width: int, start_depth: int = 1):
        super().__init__(); self.width = int(width)
        self.layers = nn.ModuleList([nn.Linear(width, width) for _ in range(max(1, start_depth))])
    @property
    def depth(self) -> int: return len(self.layers)
    def append_depth(self, n: int = 1) -> None:
        for _ in range(n): self.layers.append(nn.Linear(self.width, self.width))
    def shrink_to(self, d: int) -> None:
        while len(self.layers) > d: self.layers.pop()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1: x = F.relu(x)
        return x

# -------------------------------
# Model (2‑Head)
# -------------------------------
class MLPAlt2Head(nn.Module):
    def __init__(self, in_dim: int, out_pq: int, out_vm: int, width: int = 64, depth: int = 1):
        super().__init__()
        self.in_dim, self.width = in_dim, int(width)
        self.fc1 = nn.Linear(in_dim, width)
        self.pq_fc2 = StackedLinear(width, depth)
        self.vm_fc2 = StackedLinear(width, depth)
        self.head_pq = nn.Linear(width, out_pq)
        self.head_vm = nn.Linear(width, out_vm)
    @property
    def depth(self) -> int: return self.pq_fc2.depth
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        pq = self.pq_fc2(x); pq = F.relu(pq) if self.pq_fc2.depth > 1 else pq
        vm = self.vm_fc2(x); vm = F.relu(vm) if self.vm_fc2.depth > 1 else vm
        return torch.cat([self.head_pq(pq), self.head_vm(vm)], dim=-1)
    def arch_spec(self): return {"width": self.width, "depth": self.depth}
    def snapshot(self): return self.arch_spec(), deepcopy(self.state_dict())
    def _rebuild_width(self, neww: int) -> None:
        neww = int(neww)
        new_fc1 = nn.Linear(self.in_dim, neww); _copy_overlap_linear(self.fc1, new_fc1)
        new_pq = StackedLinear(neww, self.pq_fc2.depth); new_vm = StackedLinear(neww, self.vm_fc2.depth)
        for i in range(self.pq_fc2.depth):
            _copy_overlap_linear(self.pq_fc2.layers[i], new_pq.layers[i])
            _copy_overlap_linear(self.vm_fc2.layers[i], new_vm.layers[i])
        new_hpq = nn.Linear(neww, self.head_pq.out_features); _copy_overlap_linear(self.head_pq, new_hpq)
        new_hvm = nn.Linear(neww, self.head_vm.out_features); _copy_overlap_linear(self.head_vm, new_hvm)
        self.fc1, self.pq_fc2, self.vm_fc2, self.head_pq, self.head_vm = new_fc1, new_pq, new_vm, new_hpq, new_hvm
        self.width = neww
    def restore(self, spec, state):
        if spec["width"] != self.width: self._rebuild_width(spec["width"]) 
        d = spec["depth"]; cur = self.depth
        if d < cur: self.pq_fc2.shrink_to(d); self.vm_fc2.shrink_to(d)
        elif d > cur: self.pq_fc2.append_depth(d - cur); self.vm_fc2.append_depth(d - cur)
        self.load_state_dict(state, strict=True)
    def widen_all(self, dk: int): self._rebuild_width(self.width + int(dk))
    def append_depth(self, n: int = 1): self.pq_fc2.append_depth(n); self.vm_fc2.append_depth(n)

# -------------------------------
# Train / Eval
# -------------------------------
@torch.no_grad()
def evaluate(model: nn.Module, val_loader: Iterable,
             loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
             physics_metric: Optional[Callable[[torch.Tensor, torch.Tensor], float]] = None,
             device: Optional[torch.device] = None):
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
class ADP_ALT_WIDTH_2Head:
    def __init__(self, in_dim: int, out_pq: int, out_vm: int, width: int = 64, depth: int = 1,
                 ex_k: int = 8, max_neurons: int = 4096, max_depth: int = 16,
                 delta: float = 0.0, patience_width: int = 10, patience_depth: int = 10,
                 device: Optional[torch.device] = None):
        self.model = MLPAlt2Head(in_dim, out_pq, out_vm, width, depth)
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
            if self.model.depth >= self.max_depth:
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
