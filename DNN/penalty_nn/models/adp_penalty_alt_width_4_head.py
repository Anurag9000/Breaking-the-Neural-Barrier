"""
Penalty‑ADP — Alternating (Width→Depth) — 4‑Head
- Train with composite penalty loss; **select by validation MSE** (not penalties).
- Four branches / four heads; outputs concatenated.
- Provide `penalty_fn(pred, xb, yb)` returning (eq_term, ineq_term) tensors.
"""
from __future__ import annotations
import math
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Optional, Tuple

import torch
from torch import nn
from torch.nn import functional as F

# -------------------------------
# Blocks
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
    def append_depth(self, n: int = 1):
        for _ in range(n): self.layers.append(nn.Linear(self.width, self.width))
    def shrink_to(self, d: int):
        while len(self.layers) > d: self.layers.pop()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, l in enumerate(self.layers):
            x = l(x)
            if i < len(self.layers) - 1: x = F.relu(x)
        return x

class MLPAlt4Head(nn.Module):
    def __init__(self, in_dim: int, outs: List[int], width: int = 64, depth: int = 1):
        super().__init__(); assert len(outs) == 4
        self.in_dim, self.width = in_dim, int(width)
        self.fc1 = nn.Linear(in_dim, width)
        self.b_fc2 = nn.ModuleList([StackedLinear(width, depth) for _ in range(4)])
        self.heads = nn.ModuleList([nn.Linear(width, o) for o in outs])
    @property
    def depth(self) -> int: return self.b_fc2[0].depth
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(x))
        outs = []
        for i in range(4):
            h = self.b_fc2[i](x)
            h = F.relu(h) if self.b_fc2[i].depth > 1 else h
            outs.append(self.heads[i](h))
        return torch.cat(outs, dim=-1)
    def arch_spec(self): return {"width": self.width, "depth": self.depth}
    def snapshot(self): return self.arch_spec(), deepcopy(self.state_dict())
    def _rebuild_width(self, neww: int) -> None:
        new_fc1 = nn.Linear(self.in_dim, neww); _copy_overlap_linear(self.fc1, new_fc1)
        new_b = nn.ModuleList([StackedLinear(neww, b.depth) for b in self.b_fc2])
        for i, b in enumerate(self.b_fc2):
            for j in range(b.depth): _copy_overlap_linear(b.layers[j], new_b[i].layers[j])
        new_heads = nn.ModuleList([nn.Linear(neww, h.out_features) for h in self.heads])
        for i, h in enumerate(self.heads): _copy_overlap_linear(h, new_heads[i])
        self.fc1, self.b_fc2, self.heads = new_fc1, new_b, new_heads
        self.width = int(neww)
    def restore(self, spec, state):
        if spec["width"] != self.width: self._rebuild_width(spec["width"]) 
        d = spec["depth"]; cur = self.depth
        if d < cur:
            for b in self.b_fc2: b.shrink_to(d)
        elif d > cur:
            for b in self.b_fc2: b.append_depth(d - cur)
        self.load_state_dict(state, strict=True)
    def widen_all(self, dk: int): self._rebuild_width(self.width + int(dk))
    def append_depth(self, n: int = 1):
        for b in self.b_fc2: b.append_depth(n)

# -------------------------------
# Train / Eval
# -------------------------------
@torch.no_grad()
def evaluate_mse(model: nn.Module, val_loader: Iterable,
                 loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 device: Optional[torch.device] = None) -> float:
    model.eval(); tot, n = 0.0, 0
    for xb, yb in val_loader:
        xb = xb.to(device) if device else xb; yb = yb.to(device) if device else yb
        out = model(xb); loss = loss_fn(out, yb)
        tot += float(loss.detach()) * len(xb); n += len(xb)
        return tot / max(1, n)

def inner_train_penalty(model: nn.Module, train_loader: Iterable, val_loader: Iterable,
                        loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                        penalty_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
                        lambda_loss: float = 1.0, lambda_eq: float = 0.0, lambda_ineq: float = 0.0,
                        optimizer_ctor: Callable[[Iterable], torch.optim.Optimizer] = lambda p: torch.optim.Adam(p, lr=1e-3),
                        es_patience: int = 20, device: Optional[torch.device] = None):
    opt = optimizer_ctor(model.parameters())
    best_v = math.inf; best_state = deepcopy(model.state_dict())
    patience, epochs = es_patience, 0
    zero = None
    while True:
        epochs += 1
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device) if device else xb; yb = yb.to(device) if device else yb
            opt.zero_grad(set_to_none=True)
            pred = model(xb)
            mse = loss_fn(pred, yb)
            if zero is None:
                zero = torch.zeros((), device=pred.device)
            if penalty_fn is None:
                eq_term, ineq_term = zero, zero
            else:
                eq_term, ineq_term = penalty_fn(pred, xb, yb)
                eq_term = eq_term.mean() if eq_term.ndim > 0 else eq_term
                ineq_term = ineq_term.mean() if ineq_term.ndim > 0 else ineq_term
            total = lambda_loss * mse + lambda_eq * eq_term + lambda_ineq * ineq_term
            total.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step()
        v = evaluate_mse(model, val_loader, loss_fn, device)
        if v < best_v - 1e-12:
            best_v = v; best_state = deepcopy(model.state_dict()); patience = es_patience
        else:
            patience -= 1
            if patience <= 0: break
    model.load_state_dict(best_state)
    return best_v, best_state, epochs

# -------------------------------
# Controller (Alt‑Width, MSE‑first)
# -------------------------------
class PENALTY_ADP_ALT_WIDTH_4Head:
    def __init__(self, in_dim: int, outs: List[int], width: int = 64, depth: int = 1,
                 ex_k: int = 8, max_neurons: int = 4096, max_depth: int = 16,
                 delta: float = 0.0, patience_width: int = 10, patience_depth: int = 10,
                 lambda_loss: float = 1.0, lambda_eq: float = 0.0, lambda_ineq: float = 0.0,
                 device: Optional[torch.device] = None):
        self.model = MLPAlt4Head(in_dim, outs, width, depth)
        self.device = device
        if device: self.model.to(device)
        self.ex_k, self.max_neurons, self.max_depth = int(ex_k), int(max_neurons), int(max_depth)
        self.delta = float(delta); self.pw, self.pd = int(patience_width), int(patience_depth)
        self.lambda_loss, self.lambda_eq, self.lambda_ineq = float(lambda_loss), float(lambda_eq), float(lambda_ineq)
        self._g_best = math.inf; self._g_snap = self.model.snapshot()

    def fit(self, train_loader: Iterable, val_loader: Iterable,
            loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] = nn.MSELoss(),
            penalty_fn: Optional[Callable[[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]] = None,
            optimizer_ctor: Callable[[Iterable], torch.optim.Optimizer] = lambda p: torch.optim.Adam(p, lr=1e-3),
            max_global_epochs: int = 500, es_patience: int = 20):
        tot = 0
        base_v, _, e = inner_train_penalty(self.model, train_loader, val_loader, loss_fn, penalty_fn,
                                           self.lambda_loss, self.lambda_eq, self.lambda_ineq,
                                           optimizer_ctor, es_patience, self.device)
        tot += e; self._upd(base_v)
        fw = fd = 0
        while tot < max_global_epochs and not (fw >= self.pw and fd >= self.pd):
            # WIDTH first
            if self.model.width >= self.max_neurons:
                fw = self.pw
            else:
                pre = self.model.snapshot(); pv = base_v
                self.model.widen_all(self.ex_k)
                v, _, e = inner_train_penalty(self.model, train_loader, val_loader, loss_fn, penalty_fn,
                                              self.lambda_loss, self.lambda_eq, self.lambda_ineq,
                                              optimizer_ctor, es_patience, self.device)
                tot += e
                if v < pv - self.delta:
                    base_v, fw = v, 0; self._upd(v)
                else:
                    self.model.restore(*pre); fw += 1
            # DEPTH next
            if self.model.depth >= self.max_depth:
                fd = self.pd
            else:
                pre = self.model.snapshot(); pv = base_v
                self.model.append_depth(1)
                v, _, e = inner_train_penalty(self.model, train_loader, val_loader, loss_fn, penalty_fn,
                                              self.lambda_loss, self.lambda_eq, self.lambda_ineq,
                                              optimizer_ctor, es_patience, self.device)
                tot += e
                if v < pv - self.delta:
                    base_v, fd = v, 0; self._upd(v)
                else:
                    self.model.restore(*pre); fd += 1
        self.model.restore(*self._g_snap)
        return {"val_mse": float(self._g_best)}

    def _upd(self, v: float):
        if v < self._g_best - 1e-12:
            self._g_best = v; self._g_snap = self.model.snapshot()
