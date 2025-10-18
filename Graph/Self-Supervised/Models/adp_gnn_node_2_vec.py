# Node2Vec — biased random walks + skip-gram (single-model)

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.utils import to_undirected
except Exception:
    raise ImportError("Requires torch_geometric.")

class SkipGram(nn.Module):
    def __init__(self, num_nodes, embed_dim):
        super().__init__()
        self.input = nn.Embedding(num_nodes, embed_dim)
        self.output = nn.Embedding(num_nodes, embed_dim)

    def forward(self, u, v, neg):
        u_e = self.input(u)
        v_e = self.output(v)
        pos = (u_e * v_e).sum(dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        neg_e = self.output(neg)
        neg_score = torch.bmm(neg_e, u_e.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return pos_loss + neg_loss

@dataclass
class TrainConfig:
    lr: float = 3e-3
    weight_decay: float = 0.0
    max_epochs: int = 5
    patience: int = 2
    grad_clip: Optional[float] = None
    walk_length: int = 40
    walks_per_node: int = 5
    window_size: int = 5
    num_neg: int = 5
    p: float = 1.0
    q: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

# Precompute neighbors
@torch.no_grad()
def build_adj(edge_index: torch.Tensor, num_nodes: int):
    edge_index = to_undirected(edge_index)
    row, col = edge_index
    adj = [[] for _ in range(num_nodes)]
    for u, v in zip(row.tolist(), col.tolist()):
        adj[u].append(v)
        adj[v].append(u)
    return adj

@torch.no_grad()
def biased_walks(adj, p: float, q: float, wl: int, wpn: int) -> List[List[int]]:
    import random
    num_nodes = len(adj)
    walks = []
    for u in range(num_nodes):
        for _ in range(wpn):
            walk = [u]
            prev = -1
            cur = u
            for _ in range(wl - 1):
                neigh = adj[cur]
                if not neigh:
                    break
                # transition weights
                weights = []
                for v in neigh:
                    if v == prev: w = 1.0 / p
                    elif v in adj[prev] if prev != -1 else False: w = 1.0
                    else: w = 1.0 / q
                    weights.append(w)
                # sample proportional to weights
                s = sum(weights)
                r = random.random() * s
                acc = 0.0
                chosen = neigh[-1]
                for v, w in zip(neigh, weights):
                    acc += w
                    if r <= acc:
                        chosen = v; break
                walk.append(chosen)
                prev, cur = cur, chosen
            walks.append(walk)
    return walks

@torch.no_grad()
def skipgram_pairs(walks: List[List[int]], window: int) -> List[Tuple[int,int]]:
    pairs = []
    for w in walks:
        L = len(w)
        for i, u in enumerate(w):
            s = max(0, i - window); e = min(L, i + window + 1)
            for j in range(s, e):
                if j == i: continue
                pairs.append((u, w[j]))
    return pairs

@torch.no_grad()
def negative_samples(batch_size: int, num_nodes: int, K: int, device) -> torch.Tensor:
    return torch.randint(0, num_nodes, (batch_size, K), device=device)


def evaluate_ssl(model: nn.Module, pairs: List[Tuple[int,int]], cfg: TrainConfig, num_nodes: int) -> Tuple[float, Dict[str,float]]:
    model.eval();
    with torch.no_grad():
        idx = torch.randperm(len(pairs))[: min(1024, len(pairs))]
        u = torch.tensor([pairs[i][0] for i in idx], device=cfg.device)
        v = torch.tensor([pairs[i][1] for i in idx], device=cfg.device)
        neg = negative_samples(u.size(0), num_nodes, cfg.num_neg, cfg.device)
        loss = model(u, v, neg).item()
    return loss, {}


def train_with_early_stop(model: nn.Module, pairs: List[Tuple[int,int]], cfg: TrainConfig, num_nodes: int):
    model.to(cfg.device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss=float('inf'); best_state=None; no_improve=0; ran_epochs=0
    for epoch in range(1, cfg.max_epochs+1):
        model.train()
        perm = torch.randperm(len(pairs))
        batch_size = 2048
        for i in range(0, len(pairs), batch_size):
            sl = perm[i:i+batch_size]
            u = torch.tensor([pairs[j][0] for j in sl], device=cfg.device)
            v = torch.tensor([pairs[j][1] for j in sl], device=cfg.device)
            neg = negative_samples(u.size(0), num_nodes, cfg.num_neg, cfg.device)
            opt.zero_grad(); loss = model(u, v, neg); loss.backward(); opt.step()
        ran_epochs=epoch
        val,_ = evaluate_ssl(model, pairs, cfg, num_nodes)
        if val + 1e-12 < best_loss:
            best_loss=val; best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; no_improve=0
        else:
            no_improve+=1
        if no_improve>=cfg.patience: break
    if best_state is not None: model.load_state_dict(best_state)
    return ran_epochs, best_loss, {}


def snapshot(model: nn.Module) -> Dict:
    return {"state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}

def restore(model: nn.Module, snap: Dict):
    model.load_state_dict(snap["state"])
