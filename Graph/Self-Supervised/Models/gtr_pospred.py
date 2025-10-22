import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from torch_geometric.nn import TransformerConv
from torch_geometric.utils import to_dense_adj

def shortest_path_target(edge_index, N, src=0):
    # simple BFS distance from node 0 as proxy (single-graph)
    adj = to_dense_adj(edge_index, max_num_nodes=N)[0]
    dist = torch.full((N,), float('inf'), device=adj.device); dist[src]=0
    frontier = [src]
    while frontier:
        new=[]
        for u in frontier:
            vs = adj[u].nonzero(as_tuple=False).view(-1)
            for v in vs:
                if dist[v] > dist[u]+1:
                    dist[v] = dist[u]+1; new.append(int(v))
        frontier = new
    dist[torch.isinf(dist)] = N
    return dist

class GTR(nn.Module):
    def __init__(self, in_dim, hid=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerConv(in_dim if i==0 else hid, hid//heads, heads=heads, dropout=dropout) for i in range(layers)])
        self.pred = nn.Linear(hid, 1); self.dropout=dropout
    def forward(self, data):
        x, ei = data.x, data.edge_index
        for c in self.layers:
            x = F.relu(c(x, ei)); x = F.dropout(x, p=self.dropout, training=self.training)
        return self.pred(x).squeeze(-1)

@dataclass
class Config:
    lr:float=1e-3; epochs:int=300; patience:int=40; ckpt:str="ckpt_gtr_pospred.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

class PosPred_GTR(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__(); self.backbone=GTR(in_dim); self.cfg=cfg
    def loss(self, data):
        y = shortest_path_target(data.edge_index, data.x.size(0), src=0)
        pred = self.backbone(data)
        return F.mse_loss(pred, y)
