# GraphCL (SimCLR-style for graphs) — single-model, shared encoder for two views

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


# ------- Augmentations -------
def aug_drop_edges(data, p=0.2):
    ei, _ = dropout_edge(data.edge_index, p=p)
    data2 = data.clone()
    data2.edge_index = ei
    return data2

def aug_mask_features(data, p=0.2):
    x = data.x
    mask = torch.rand_like(x) < p
    x2 = x.clone()
    x2[mask] = 0.0
    data2 = data.clone()
    data2.x = x2
    return data2

# ------- Encoder -------
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


class GraphCL(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, proj_dim=128, num_layers=3, dropout=0.2, tau=0.2):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden, out_dim, num_layers, dropout)
        self.proj = ProjectionHead(out_dim, proj_dim)
        self.tau = tau

    def graph_embed(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        z = self.encoder(x, edge_index)
        g = global_add_pool(z, batch)
        g = F.normalize(self.proj(g), dim=1)
        return g

    def nt_xent(self, z1, z2):
        z = torch.cat([z1, z2], dim=0)
        sim = F.cosine_similarity(z.unsqueeze(1), z.unsqueeze(0), dim=-1)  # [2B,2B]
        sim = sim / self.tau
        B = z1.size(0)
        labels = torch.arange(B, device=z1.device)
        labels = torch.cat([labels + B, labels], dim=0)
        mask = torch.eye(2*B, device=z1.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        loss = F.cross_entropy(sim, labels)
        return loss

    def forward(self, data):
        # Create two stochastic augmentations with shared encoder
        d1 = aug_drop_edges(aug_mask_features(data, p=0.2), p=0.2)
        d2 = aug_drop_edges(aug_mask_features(data, p=0.2), p=0.2)
        z1 = self.graph_embed(d1)
        z2 = self.graph_embed(d2)
        loss = self.nt_xent(z1, z2)
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
    model.eval()
    with torch.no_grad():
        loss = model(data).item()
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
        loss = model(data)
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
