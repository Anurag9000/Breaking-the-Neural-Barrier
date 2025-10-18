# GRACE — Graph Contrastive Encoding (single-model, shared encoder for two stoch. views)

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.nn import GCNConv
    from torch_geometric.utils import dropout_edge
except Exception:
    raise ImportError("Requires torch_geometric.")


def aug_feature_mask(x, p=0.2):
    mask = torch.rand_like(x) < p
    x2 = x.clone()
    x2[mask] = 0.0
    return x2

class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden, hidden))
        self.convs.append(GCNConv(hidden, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GRACE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.2, tau=0.4):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden, out_dim, num_layers, dropout)
        self.tau = tau

    def embed(self, data, feat_p=0.2, edge_p=0.2):
        x, edge_index = data.x, data.edge_index
        x2 = aug_feature_mask(x, feat_p)
        ei2, _ = dropout_edge(edge_index, p=edge_p)
        z = self.encoder(x2, ei2)
        z = F.normalize(z, dim=1)
        return z

    def nt_xent_nodes(self, z1, z2):
        # Node-level contrast; positives are same-node across views
        z = torch.cat([z1, z2], dim=0)
        sim = torch.mm(z, z.t()) / self.tau
        N = z1.size(0)
        labels = torch.arange(N, device=z1.device)
        labels = torch.cat([labels + N, labels], dim=0)
        mask = torch.eye(2*N, device=z1.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        loss = F.cross_entropy(sim, labels)
        return loss

    def forward(self, data):
        z1 = self.embed(data)
        z2 = self.embed(data)
        loss = self.nt_xent_nodes(z1, z2)
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
