
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

def load_planetoid(name: str = "Cora", root: str = "./data"):
    try:
        from torch_geometric.datasets import Planetoid
    except Exception as e:
        raise ImportError("Requires torch_geometric. Install: pip install torch-geometric (plus deps).") from e
    ds = Planetoid(root=root, name=name)
    data = ds[0]
    return (data.x, data.y, data.train_mask, data.val_mask, data.test_mask), ds.num_classes

class DNNNodeFC(nn.Module):
    def __init__(self, num_nodes: int, num_classes: int, hidden: int = 128, depth: int = 2):
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

    @torch.no_grad()
    def total_neurons(self) -> int:
        return self.hidden * (self.depth + 1) + self.num_nodes * self.num_classes

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        N, Fdim = X.shape
        assert N == self.num_nodes
        self._ensure_feature_proj(Fdim)
        z = (X @ self.feature_proj).unsqueeze(0)
        z = F.relu(self.in_lin(z))
        for lin in self.hiddens:
            z = F.relu(lin(z))
        z = self.out_lin(z).view(N, self.num_classes)
        return z

@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 5e-4
    max_epochs: int = 2000
    patience: int = 100
    grad_clip: float = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

def train_nodeclf(model: DNNNodeFC, data, cfg: TrainCfg):
    X, y, train_mask, val_mask, test_mask = data
    X = X.to(cfg.device); y = y.to(cfg.device)
    train_mask = train_mask.to(cfg.device); val_mask = val_mask.to(cfg.device); test_mask = test_mask.to(cfg.device)
    model = model.to(cfg.device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float("inf"); best_state=None; best_epoch=-1; patience=cfg.patience
    for epoch in range(1, cfg.max_epochs+1):
        model.train(); opt.zero_grad()
        logits = model(X)
        loss = F.cross_entropy(logits[train_mask], y[train_mask])
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        model.eval()
        with torch.no_grad():
            logits = model(X)
            val_loss = F.cross_entropy(logits[val_mask], y[val_mask]).item()
            if val_loss < best_val - 1e-9:
                best_val = val_loss; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; best_epoch=epoch; patience=cfg.patience
            else:
                patience -= 1
        if patience <= 0: break
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logits = model(X)
        test_acc = (logits[test_mask].argmax(-1) == y[test_mask]).float().mean().item()
    return best_val, test_acc, best_epoch

def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Cora", choices=["Cora","Citeseer","PubMed"])
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--depth", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=100)
    args = p.parse_args()
    data, num_classes = load_planetoid(args.dataset)
    X, y, train_mask, val_mask, test_mask = data
    N = X.size(0)
    model = DNNNodeFC(N, num_classes, hidden=args.hidden, depth=args.depth)
    cfg = TrainCfg(lr=args.lr, patience=args.patience)
    best_val, test_acc, best_epoch = train_nodeclf(model, data, cfg)
    print(f"[STL] Dataset={args.dataset} N={N} C={num_classes} H={args.hidden} D={args.depth}")
    print(f"Best Val Loss={best_val:.4f} at epoch {best_epoch}, Test@1={test_acc*100:.2f}%")
