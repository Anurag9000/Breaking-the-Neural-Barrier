# Edge Prediction pretext — mask a subset of edges and predict their existence

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import negative_sampling
except Exception:
    raise ImportError("Requires torch_geometric.")

class EdgePred(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_dim))
        self.dropout = dropout
        self.edge_mlp = nn.Sequential(nn.Linear(2*out_dim, out_dim), nn.ReLU(), nn.Linear(out_dim, 1))

    def encode(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

    def score_edges(self, z, edge_index):
        src, dst = edge_index
        pair = torch.cat([z[src], z[dst]], dim=1)
        return self.edge_mlp(pair).squeeze(-1)

    def forward(self, data, sample_ratio=0.3):
        x, edge_index, num_nodes = data.x, data.edge_index, data.num_nodes
        z = self.encode(x, edge_index)
        # subsample positive edges
        E = edge_index.size(1)
        k = max(1, int(E * sample_ratio))
        perm = torch.randperm(E, device=edge_index.device)[:k]
        pos_e = edge_index[:, perm]
        # sample negatives equal to positives
        neg_e = negative_sampling(pos_e, num_nodes=num_nodes, num_neg_samples=k)
        pos_s = self.score_edges(z, pos_e)
        neg_s = self.score_edges(z, neg_e)
        pos_loss = F.binary_cross_entropy_with_logits(pos_s, torch.ones_like(pos_s))
        neg_loss = F.binary_cross_entropy_with_logits(neg_s, torch.zeros_like(neg_s))
        return pos_loss + neg_loss


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 60
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
    best_loss=float('inf'); best_state=None; no_improve=0; ran_epochs=0
    for epoch in range(1, cfg.max_epochs+1):
        model.train(); opt.zero_grad(); loss = model(data); loss.backward()
        if cfg.grad_clip is not None: nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        val,_ = evaluate_ssl(model, data); ran_epochs=epoch
        if val + 1e-12 < best_loss:
            best_loss=val; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; no_improve=0
        else: no_improve+=1
        if no_improve>=cfg.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return ran_epochs, best_loss, {}


def snapshot(model: nn.Module) -> Dict:
    return {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

def restore(model: nn.Module, snap: Dict):
    model.load_state_dict(snap["state"])
