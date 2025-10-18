# GraphMAE — masked node feature reconstruction with a single encoder + lightweight decoder

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.nn import GATConv
except Exception:
    raise ImportError("Requires torch_geometric.")


def mask_node_features(x, mask_ratio=0.3):
    N, D = x.size()
    mask = torch.rand(N, device=x.device) < mask_ratio
    x_masked = x.clone()
    x_masked[mask] = 0.0
    return x_masked, mask

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GATConv(in_dim, hidden, heads=heads, dropout=dropout, concat=True))
        for _ in range(num_layers - 2):
            self.layers.append(GATConv(hidden*heads, hidden, heads=heads, dropout=dropout, concat=True))
        self.layers.append(GATConv(hidden*heads, out_dim, heads=1, dropout=dropout, concat=False))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.elu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class GraphMAE(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, heads=4, dropout=0.2):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden, out_dim, num_layers, heads, dropout)
        self.decoder = nn.Sequential(
            nn.Linear(out_dim, hidden), nn.ReLU(), nn.Linear(hidden, in_dim)
        )

    def forward(self, data, mask_ratio=0.3):
        x, edge_index = data.x, data.edge_index
        x_masked, mask = mask_node_features(x, mask_ratio)
        z = self.encoder(x_masked, edge_index)
        x_rec = self.decoder(z)
        loss = F.mse_loss(x_rec[mask], x[mask]) if mask.any() else (x_rec - x).pow(2).mean()
        return loss


@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 300
    patience: int = 80
    grad_clip: Optional[float] = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_ssl(model: nn.Module, data) -> Tuple[float, Dict[str, float]]:
    model.eval()
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
