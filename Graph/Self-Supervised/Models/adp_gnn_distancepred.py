# Distance / Position Prediction — predict shortest-path distance bins to anchors

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.nn import GCNConv
except Exception:
    raise ImportError("Requires torch_geometric.")


def compute_anchor_targets(edge_index, num_nodes, num_anchors=8, max_bin=4):
    # Simple BFS-based distances from random anchors; clamp to max_bin and one-hot encode
    device = edge_index.device
    anchors = torch.randperm(num_nodes, device=device)[:num_anchors]
    row, col = edge_index
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(row.tolist(), col.tolist()):
        adj[u].append(v)
        adj[v].append(u)
    dists = torch.full((num_nodes, num_anchors), max_bin, device=device, dtype=torch.long)
    for j, a in enumerate(anchors.tolist()):
        # BFS
        from collections import deque
        q = deque([a])
        seen = {a: 0}
        while q:
            u = q.popleft()
            for w in adj[u]:
                if w not in seen:
                    seen[w] = seen[u] + 1
                    q.append(w)
        for n, dist in seen.items():
            dists[n, j] = min(dist, max_bin)
    # convert to one-hot (max_bin+1 classes)
    K = max_bin + 1
    targets = F.one_hot(dists, num_classes=K).float()  # [N, A, K]
    return targets, anchors

class Encoder(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.layers.append(GCNConv(in_dim, hidden))
        for _ in range(num_layers - 2):
            self.layers.append(GCNConv(hidden, hidden))
        self.layers.append(GCNConv(hidden, out_dim))
        self.dropout = dropout

    def forward(self, x, edge_index):
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if i < len(self.layers) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        return x

class DistancePred(nn.Module):
    def __init__(self, in_dim, hidden, out_dim, num_layers=2, dropout=0.2, num_anchors=8, max_bin=4):
        super().__init__()
        self.encoder = Encoder(in_dim, hidden, out_dim, num_layers, dropout)
        self.num_anchors = num_anchors
        self.max_bin = max_bin
        K = max_bin + 1
        self.head = nn.Linear(out_dim, self.num_anchors * K)
        self.K = K

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        z = self.encoder(x, edge_index)
        logits = self.head(z)  # [N, A*K]
        logits = logits.view(z.size(0), self.num_anchors, self.K)
        with torch.no_grad():
            targets, anchors = compute_anchor_targets(edge_index, data.num_nodes, self.num_anchors, self.max_bin)
        loss = F.cross_entropy(logits.permute(0,2,1), targets.argmax(dim=-1))  # CE over K for each anchor
        return loss


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
