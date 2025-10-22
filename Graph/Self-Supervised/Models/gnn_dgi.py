# Single-model DGI for node-level SSL (PyTorch Geometric)
# Style aligned with ADP CNN files: Config dataclass, evaluate/train_with_early_stop, snapshot/restore

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.nn import GCNConv, global_mean_pool
    from torch_geometric.utils import shuffle_node
except Exception as e:
    raise ImportError("Requires torch_geometric. Install PyG to use this file.")


# ------------------------------
# Model components
# ------------------------------
class GCNEncoder(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        assert num_layers >= 2
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

class Readout(nn.Module):
    def forward(self, z):
        # Global summary vector (mean followed by nonlinearity)
        s = torch.sigmoid(z.mean(dim=0, keepdim=True))
        return s

class Discriminator(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.bilinear = nn.Bilinear(dim, dim, 1)

    def forward(self, s, z):
        # s: [1, d], z: [N, d]
        s_exp = s.expand(z.size(0), -1)
        return self.bilinear(z, s_exp).squeeze(-1)


class DGI(nn.Module):
    def __init__(self, in_dim: int, hidden: int, out_dim: int, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.encoder = GCNEncoder(in_dim, hidden, out_dim, num_layers, dropout)
        self.readout = Readout()
        self.disc = Discriminator(out_dim)

    def corruption(self, x):
        # Shuffle features along node dimension for negative samples
        perm = torch.randperm(x.size(0), device=x.device)
        return x[perm]

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encoder(x, edge_index)                 # positive
        s = self.readout(z)                              # summary
        x_corrupt = self.corruption(x)
        z_tilde = self.encoder(x_corrupt, edge_index)    # negative (shared encoder)
        pos = self.disc(s, z)
        neg = self.disc(s, z_tilde)
        return z, s, pos, neg

    def ssl_loss(self, pos, neg):
        # DGI uses a binary logistic objective
        pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        neg_loss = F.binary_cross_entropy_with_logits(neg, torch.zeros_like(neg))
        return pos_loss + neg_loss


# ------------------------------
# Training utilities (ADP style)
# ------------------------------
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
        _, _, pos, neg = model(data)
        loss = model.ssl_loss(pos, neg).item()
        # AUC-like proxy: sigmoid(pos) should be high, sigmoid(neg) low
        pos_m = torch.sigmoid(pos).mean().item()
        neg_m = torch.sigmoid(neg).mean().item()
    return loss, {"pos_score": pos_m, "neg_score": neg_m}


def train_with_early_stop(model: nn.Module, data, cfg: TrainConfig) -> Tuple[int, float, Dict[str, float]]:
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
        _, _, pos, neg = model(data)
        loss = model.ssl_loss(pos, neg)
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        val_loss, _ = evaluate_ssl(model, data)  # self-supervised; validate on same graph
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


# ------------------------------
# Snapshot helpers (for parity with ADP style)
# ------------------------------
def snapshot(model: nn.Module) -> Dict:
    return {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}


def restore(model: nn.Module, snap: Dict):
    model.load_state_dict(snap["state"])
