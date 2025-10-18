
import time, math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from dnn_stl_graph import load_planetoid
from adp_dnn_resize import _resize_linear

class AdaptiveDNNNodeFC(nn.Module):
    def __init__(self, num_nodes: int, num_classes: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        assert depth >= 1
        self.num_nodes = num_nodes
        self.num_classes = num_classes
        self.hidden = hidden
        self.depth = depth
        self.feature_proj = None
        self.in_lin = nn.Linear(num_nodes, hidden, bias=False)
        self.hiddens = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(depth-1)])
        self.out_lin = nn.Linear(hidden, num_nodes * num_classes, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.in_lin.weight, nonlinearity="relu")
        for lin in self.hiddens:
            nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
        nn.init.zeros_(self.out_lin.bias)
        nn.init.kaiming_normal_(self.out_lin.weight, nonlinearity="linear")

    def _ensure_feature_proj(self, Fdim: int):
        if self.feature_proj is None:
            self.feature_proj = nn.Parameter(torch.zeros(Fdim))
            nn.init.normal_(self.feature_proj, std=0.02)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N, Fdim = X.shape
        assert N == self.num_nodes
        self._ensure_feature_proj(Fdim)
        z = (X @ self.feature_proj).unsqueeze(0)
        z = torch.relu(self.in_lin(z))
        for lin in self.hiddens:
            z = torch.relu(lin(z))
        z = self.out_lin(z).view(N, self.num_classes)
        return z

    @torch.no_grad()
    def total_neurons(self) -> int:
        return self.hidden * (self.depth + 1) + self.num_nodes * self.num_classes

    def widen_all(self, delta: int):
        assert delta >= 1
        new_h = self.hidden + delta
        self.in_lin = _resize_linear(self.in_lin, in_features=self.num_nodes, out_features=new_h)
        new_hiddens = torch.nn.ModuleList()
        for lin in self.hiddens:
            new_hiddens.append(_resize_linear(lin, in_features=new_h, out_features=new_h))
        self.hiddens = new_hiddens
        self.out_lin = _resize_linear(self.out_lin, in_features=new_h, out_features=self.num_nodes * self.num_classes)
        self.hidden = new_h

    def append_depth(self):
        new_lin = nn.Linear(self.hidden, self.hidden, bias=False)
        nn.init.kaiming_normal_(new_lin.weight, nonlinearity="relu")
        self.hiddens.append(new_lin)
        self.depth += 1

    def snapshot(self):
        return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}

    def restore(self, st):
        self.load_state_dict(st, strict=True)

@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 5e-4
    max_epochs: int = 1000
    patience: int = 100
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_early_stop(model: AdaptiveDNNNodeFC, data, cfg: TrainCfg):
    X, y, train_mask, val_mask, test_mask = data
    X = X.to(cfg.device); y = y.to(cfg.device)
    train_mask = train_mask.to(cfg.device); val_mask = val_mask.to(cfg.device); test_mask = test_mask.to(cfg.device)
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf"); best_state=None; best_epoch=-1; patience=cfg.patience
    for epoch in range(1, cfg.max_epochs+1):
        model.train(); opt.zero_grad()
        logits = model(X)
        loss = torch.nn.functional.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        if cfg.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(X)
            val_loss = torch.nn.functional.cross_entropy(logits[val_mask], y[val_mask]).item()
            if val_loss < best_val - 1e-9:
                best_val = val_loss; best_state = model.snapshot(); best_epoch=epoch; patience=cfg.patience
            else:
                patience -= 1
        if patience <= 0: break
    if best_state is not None:
        model.restore(best_state)
    with torch.no_grad():
        logits = model(X)
        test_acc = (logits[test_mask].argmax(-1) == y[test_mask]).float().mean().item()
    return best_val, test_acc, best_epoch
