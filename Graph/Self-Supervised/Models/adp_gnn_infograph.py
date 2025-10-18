# InfoGraph (single-model) in PyTorch Geometric
# MI between graph-/subgraph-/node-level reps; here: node↔graph MI objective

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.nn import GINConv, global_add_pool
except Exception:
    raise ImportError("Requires torch_geometric.")


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

class InfoGraph(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=3, dropout=0.2):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden, out_dim, num_layers, dropout)
        self.readout = lambda z, batch: global_add_pool(z, batch)  # graph summary
        self.disc = nn.Bilinear(out_dim, out_dim, 1)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        z = self.encoder(x, edge_index)
        g = self.readout(z, batch)  # [B, d]
        # normalize summaries per-graph
        g = torch.sigmoid(g)
        return z, g, batch

    def ssl_loss(self, z, g, batch):
        # Positive: node with its own graph summary; Negative: node with a different graph summary
        # Build scores for all node-graph pairs in the mini-batch
        g_exp = g[batch]  # [N, d]
        pos = self.disc(z, g_exp).squeeze(-1)
        # For negatives, roll graph indices
        neg_g = g[batch.roll(1)]
        neg = self.disc(z, neg_g).squeeze(-1)
        pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        neg_loss = F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        return pos_loss + neg_loss


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 50
    grad_clip: Optional[float] = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_ssl(model: nn.Module, data) -> Tuple[float, Dict[str, float]]:
    model.eval()
    with torch.no_grad():
        z, g, batch = model(data)
        loss = model.ssl_loss(z, g, batch).item()
    return loss, {}


def train_with_early_stop(model: nn.Module, data, cfg: TrainConfig):
    model.to(cfg.device)
    data = data.to(cfg.device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_loss = float('inf')
    best_state = None
    no_improve = 0
    ran_epochs = 0

    for epoch in range(1, cfg.max_epochs + 1):
        model.train()
        opt.zero_grad()
        z, g, batch = model(data)
        loss = model.ssl_loss(z, g, batch)
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        val_loss, _ = evaluate_ssl(model, data)
        ran_epochs = epoch
        if val_loss + 1e-12 < best_loss:
            best_loss = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return ran_epochs, best_loss, {}


def snapshot(model: nn.Module) -> Dict:
    return {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def restore(model: nn.Module, snap: Dict):
    model.load_state_dict(snap["state"])
