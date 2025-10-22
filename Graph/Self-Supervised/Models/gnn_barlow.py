# Barlow Twins for Graphs — single-model, shared encoder with cross-correlation objective

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.nn import GINConv, global_add_pool
    from torch_geometric.utils import dropout_edge
except Exception:
    raise ImportError("Requires torch_geometric.")


def aug_mask_features(x, p=0.2):
    mask = torch.rand_like(x) < p
    x2 = x.clone(); x2[mask] = 0.0
    return x2

class GINEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=3, dropout=0.2):
        super().__init__()
        def mlp(i, o):
            return nn.Sequential(nn.Linear(i, hidden), nn.ReLU(), nn.Linear(hidden, o))
        self.layers = nn.ModuleList()
        self.layers.append(GINConv(mlp(in_dim, hidden)))
        for _ in range(num_layers - 2):
            self.layers.append(GINConv(mlp(hidden, hidden)))
        self.layers.append(GINConv(mlp(hidden, out_dim)))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.layers):
            x = conv(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class ProjectionHead(nn.Module):
    def __init__(self, dim, proj_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, proj_dim)
        )
    def forward(self, z):
        return self.net(z)

class BarlowG(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, proj_dim=128, num_layers=3, dropout=0.2, lambda_offdiag=5e-3):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden, out_dim, num_layers, dropout)
        self.proj = ProjectionHead(out_dim, proj_dim)
        self.lambda_offdiag = lambda_offdiag

    def graph_embed(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        z = self.encoder(x, edge_index)
        g = global_add_pool(z, batch)
        g = self.proj(g)
        # BN on batch without running stats (normalize dims)
        g = g - g.mean(dim=0, keepdim=True)
        g = g / (g.std(dim=0, keepdim=True) + 1e-9)
        return g

    def forward(self, data):
        # two stoch. views
        d1 = data.clone(); d1.x = aug_mask_features(data.x, p=0.2); d1.edge_index, _ = dropout_edge(data.edge_index, p=0.2)
        d2 = data.clone(); d2.x = aug_mask_features(data.x, p=0.2); d2.edge_index, _ = dropout_edge(data.edge_index, p=0.2)
        z1 = self.graph_embed(d1)
        z2 = self.graph_embed(d2)
        N, D = z1.size(0), z1.size(1)
        c = (z1.T @ z2) / N  # D x D
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
        loss = on_diag + self.lambda_offdiag * off_diag
        return loss


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 50
    grad_clip: Optional[float] = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_ssl(model: nn.Module, data) -> Tuple[float, Dict[str, float]]:
    model.eval();
    with torch.no_grad():
        loss = model(data).item()
    return loss, {}


def train_with_early_stop(model: nn.Module, data, cfg: TrainConfig):
    model.to(cfg.device); data = data.to(cfg.device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss = float('inf'); best_state=None; no_improve=0; ran_epochs=0
    for epoch in range(1, cfg.max_epochs+1):
        model.train(); opt.zero_grad()
        loss = model(data); loss.backward()
        if cfg.grad_clip is not None: nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        val, _ = evaluate_ssl(model, data); ran_epochs=epoch
        if val + 1e-12 < best_loss:
            best_loss = val; best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; no_improve=0
        else: no_improve += 1
        if no_improve >= cfg.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return ran_epochs, best_loss, {}


def snapshot(model: nn.Module) -> Dict:
    return {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

def restore(model: nn.Module, snap: Dict):
    model.load_state_dict(snap["state"])
