# DeepCluster-G — k-means pseudo-labeling + classifier fine-tuning (single-model)

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

# -----------------
# Encoder + Head
# -----------------
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

class DeepClusterG(nn.Module):
    def __init__(self, in_dim, hidden, rep_dim, num_clusters, num_layers=3, dropout=0.2):
        super().__init__()
        self.encoder = GINEncoder(in_dim, hidden, rep_dim, num_layers, dropout)
        self.classifier = nn.Linear(rep_dim, num_clusters)
        self.num_clusters = num_clusters

    def embed_graph(self, data):
        x, edge_index, batch = data.x, data.edge_index, getattr(data, 'batch', None)
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)
        z = self.encoder(x, edge_index)
        g = global_add_pool(z, batch)
        g = F.normalize(g, dim=1)
        return g

    def forward(self, data):
        g = self.embed_graph(data)
        logits = self.classifier(g)
        return logits

# -----------------
# Simple k-means (torch)
# -----------------
@torch.no_grad()
def kmeans(x: torch.Tensor, k: int, iters: int = 20) -> Tuple[torch.Tensor, torch.Tensor]:
    # x: [B, D]
    B, D = x.shape
    device = x.device
    # init with random samples
    idx = torch.randperm(B, device=device)[:k]
    c = x[idx]
    for _ in range(iters):
        # assign
        dist = torch.cdist(x, c)
        y = dist.argmin(dim=1)
        # update
        for i in range(k):
            m = (y == i)
            if m.any():
                c[i] = x[m].mean(dim=0)
    return y, c

# -----------------
# Training utils
# -----------------
@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-2
    max_epochs: int = 200
    patience: int = 30
    grad_clip: Optional[float] = 1.0
    reassign_every: int = 5  # epochs between k-means refresh
    kmeans_iters: int = 20
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def evaluate_ssl(model: nn.Module, data) -> Tuple[float, Dict[str, float]]:
    model.eval();
    with torch.no_grad():
        g = model.embed_graph(data)
        logits = model.classifier(g)
        # entropy of assignments as a proxy
        probs = logits.softmax(dim=1)
        ent = (-probs * (probs.clamp_min(1e-9).log())).sum(dim=1).mean().item()
    return ent, {"avg_entropy": ent}


def train_with_early_stop(model: nn.Module, data, cfg: TrainConfig):
    model.to(cfg.device); data = data.to(cfg.device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_metric = float('inf')
    best_state = None
    no_improve = 0
    ran_epochs = 0

    # initial pseudo-labels
    labels, _ = kmeans(model.embed_graph(data), model.num_clusters, cfg.kmeans_iters)

    for epoch in range(1, cfg.max_epochs + 1):
        model.train(); opt.zero_grad()
        logits = model(data)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        if cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()

        if epoch % cfg.reassign_every == 0:
            labels, _ = kmeans(model.embed_graph(data), model.num_clusters, cfg.kmeans_iters)

        # use entropy proxy as "val"; lower entropy suggests confident clustering
        val_metric, _ = evaluate_ssl(model, data)
        ran_epochs = epoch
        if val_metric + 1e-12 < best_metric:
            best_metric = val_metric
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
        if no_improve >= cfg.patience:
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return ran_epochs, best_metric, {}


def snapshot(model: nn.Module) -> Dict:
    return {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

def restore(model: nn.Module, snap: Dict):
    model.load_state_dict(snap["state"])
