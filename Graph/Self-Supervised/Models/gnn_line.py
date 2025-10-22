# LINE — first- and second-order proximity (single-model)

from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW

try:
    from torch_geometric.utils import to_undirected
except Exception:
    raise ImportError("Requires torch_geometric.")

class LINE(nn.Module):
    def __init__(self, num_nodes, embed_dim, order='both'):
        super().__init__()
        self.order = order
        self.emb1 = nn.Embedding(num_nodes, embed_dim)
        self.ctx1 = nn.Embedding(num_nodes, embed_dim)
        if order in ('second', 'both'):
            self.emb2 = nn.Embedding(num_nodes, embed_dim)
            self.ctx2 = nn.Embedding(num_nodes, embed_dim)

    def first_order_loss(self, u, v, neg):
        u_e = self.emb1(u); v_e = self.ctx1(v)
        pos = (u_e * v_e).sum(dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        neg_e = self.ctx1(neg)
        neg_score = torch.bmm(neg_e, u_e.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return pos_loss + neg_loss

    def second_order_loss(self, u, v, neg):
        u_e = self.emb2(u); v_e = self.ctx2(v)
        pos = (u_e * v_e).sum(dim=1)
        pos_loss = F.binary_cross_entropy_with_logits(pos, torch.ones_like(pos))
        neg_e = self.ctx2(neg)
        neg_score = torch.bmm(neg_e, u_e.unsqueeze(-1)).squeeze(-1)
        neg_loss = F.binary_cross_entropy_with_logits(neg_score, torch.zeros_like(neg_score))
        return pos_loss + neg_loss

    def forward(self, u, v, neg):
        loss = self.first_order_loss(u, v, neg)
        if hasattr(self, 'emb2'):
            loss = loss + self.second_order_loss(u, v, neg)
        return loss

@dataclass
class TrainConfig:
    lr: float = 3e-3
    weight_decay: float = 0.0
    max_epochs: int = 5
    patience: int = 2
    grad_clip: Optional[float] = None
    window_size: int = 5
    num_neg: int = 5
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@torch.no_grad()
def edges_to_pairs(edge_index: torch.Tensor) -> torch.Tensor:
    # Return edge pairs as skip-gram positives
    edge_index = to_undirected(edge_index)
    return edge_index.t()  # [E,2]

@torch.no_grad()
def negative_samples(batch_size: int, num_nodes: int, K: int, device) -> torch.Tensor:
    return torch.randint(0, num_nodes, (batch_size, K), device=device)


def evaluate_ssl(model: nn.Module, pairs: torch.Tensor, cfg: TrainConfig, num_nodes: int) -> Tuple[float, Dict[str,float]]:
    model.eval();
    with torch.no_grad():
        idx = torch.randperm(pairs.size(0))[: min(2048, pairs.size(0))]
        u = pairs[idx,0].to(cfg.device); v = pairs[idx,1].to(cfg.device)
        neg = negative_samples(u.size(0), num_nodes, cfg.num_neg, cfg.device)
        loss = model(u, v, neg).item()
    return loss, {}


def train_with_early_stop(model: nn.Module, pairs: torch.Tensor, cfg: TrainConfig, num_nodes: int):
    model.to(cfg.device)
    opt = AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_loss=float('inf'); best_state=None; no_improve=0; ran_epochs=0
    batch_size = 4096
    for epoch in range(1, cfg.max_epochs+1):
        model.train()
        perm = torch.randperm(pairs.size(0))
        for i in range(0, pairs.size(0), batch_size):
            sl = perm[i:i+batch_size]
            u = pairs[sl,0].to(cfg.device); v = pairs[sl,1].to(cfg.device)
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
