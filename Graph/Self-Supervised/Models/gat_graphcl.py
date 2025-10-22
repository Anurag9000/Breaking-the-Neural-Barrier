import math, random, os, torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.utils import dropout_adj
from torch_geometric.data import Data

def set_seed(s=42):
    random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

# --------- Graph augmentations (GraphCL) ----------
def aug_edge_drop(data, p=0.2):
    edge_index, _ = dropout_adj(data.edge_index, p=p, force_undirected=True)
    d = Data(x=data.x, edge_index=edge_index, y=data.y)
    if hasattr(data, 'batch'): d.batch = data.batch
    return d

def aug_node_drop(data, p=0.2):
    keep = torch.rand(data.x.size(0), device=data.x.device) > p
    idx = keep.nonzero(as_tuple=False).view(-1)
    x = data.x[idx]
    # reindex edges
    old2new = -torch.ones(data.x.size(0), dtype=torch.long, device=idx.device)
    old2new[idx] = torch.arange(idx.size(0), device=idx.device)
    ei = data.edge_index
    mask = keep[ei[0]] & keep[ei[1]]
    ei = old2new[ei[:, mask]]
    d = Data(x=x, edge_index=ei.t().contiguous(), y=data.y)
    if hasattr(data, 'batch'):
        d.batch = data.batch[idx]
    return d

def aug_feat_mask(data, p=0.2):
    x = data.x.clone()
    mask = (torch.rand_like(x) < p).float()
    x = x * (1.0 - mask)
    d = Data(x=x, edge_index=data.edge_index, y=data.y)
    if hasattr(data, 'batch'): d.batch = data.batch
    return d

def aug_subgraph(data, ratio=0.8):
    N = data.x.size(0); k = max(2, int(N*ratio))
    idx = torch.randperm(N)[:k].to(data.x.device)
    x = data.x[idx]
    old2new = -torch.ones(N, dtype=torch.long, device=idx.device)
    old2new[idx] = torch.arange(idx.size(0), device=idx.device)
    ei = data.edge_index
    mask = (old2new[ei[0]]>=0) & (old2new[ei[1]]>=0)
    ei = torch.stack([old2new[ei[0,mask]], old2new[ei[1,mask]]], dim=0)
    d = Data(x=x, edge_index=ei, y=data.y)
    if hasattr(data, 'batch'):
        # fall back to single-graph training (Planetoid)
        d.batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    return d

AUGS = [aug_edge_drop, aug_node_drop, aug_feat_mask, aug_subgraph]

# --------- NT-Xent (SimCLR) ----------
def nt_xent(z1, z2, t=0.2):
    z1 = F.normalize(z1, dim=-1); z2 = F.normalize(z2, dim=-1)
    N = z1.size(0)
    reps = torch.cat([z1, z2], dim=0)                        # [2N, D]
    sim = reps @ reps.t()                                     # cosine (after norm)
    mask = torch.eye(2*N, device=sim.device).bool()
    sim = sim.masked_fill(mask, -9e15)
    pos = torch.sum(z1*z2, dim=-1)                            # [N]
    pos = torch.cat([pos, pos], dim=0)                        # [2N]
    sim = sim / t
    labels = torch.arange(N, device=sim.device)
    labels = torch.cat([labels+N, labels], dim=0)             # positives index
    loss = F.cross_entropy(sim, labels)
    return loss

# --------- GAT encoder ----------
class GATEncoder(nn.Module):
    def __init__(self, in_dim, hid=128, out=128, heads=4, nlayers=2, dropout=0.2):
        super().__init__()
        self.gats = nn.ModuleList()
        last = in_dim
        for i in range(nlayers):
            self.gats.append(GATv2Conv(last, hid//heads, heads=heads, dropout=dropout))
            last = hid
        self.proj = nn.Linear(last, out)
        self.dropout = dropout

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        for conv in self.gats:
            x = F.elu(conv(x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
        if hasattr(data, 'batch'):
            g = global_mean_pool(x, data.batch)
        else:
            g = x.mean(dim=0, keepdim=True)
        z = self.proj(g)
        return z

@dataclass
class Config:
    seed:int=42
    hidden:int=128
    proj:int=128
    heads:int=4
    nlayers:int=2
    dropout:float=0.2
    temperature:float=0.2
    lr:float=1e-3
    epochs:int=400
    patience:int=50
    ckpt:str="ckpt_gat_graphcl.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class GraphCL_GAT(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__()
        self.enc = GATEncoder(in_dim, cfg.hidden, cfg.proj, cfg.heads, cfg.nlayers, cfg.dropout)
        self.cfg = cfg

    def two_views(self, data):
        a1 = random.choice(AUGS); a2 = random.choice(AUGS)
        return a1(data), a2(data)

    def contrastive_loss(self, data):
        v1, v2 = self.two_views(data)
        z1 = self.enc(v1); z2 = self.enc(v2)
        return nt_xent(z1, z2, self.cfg.temperature)

    def forward(self, data):
        return self.enc(data)
