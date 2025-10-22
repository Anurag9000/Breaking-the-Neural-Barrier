import torch, torch.nn as nn, torch.nn.functional as F
from dataclasses import dataclass
from torch_geometric.nn import TransformerConv

class Encoder(nn.Module):
    def __init__(self, in_dim, hid=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerConv(in_dim if i==0 else hid, hid//heads, heads=heads, dropout=dropout) for i in range(layers)])
        self.norm = nn.LayerNorm(hid); self.dropout=dropout
    def forward(self, x, ei):
        for c in self.layers:
            x = F.relu(c(x, ei)); x = F.dropout(x, p=self.dropout, training=self.training)
        return self.norm(x)

class GraphMAE(nn.Module):
    def __init__(self, in_dim, hid=128):
        super().__init__()
        self.enc = Encoder(in_dim, hid)
        self.decoder = nn.Sequential(nn.Linear(hid, hid), nn.ReLU(), nn.Linear(hid, in_dim))
    def forward(self, data, mask):
        h = self.enc(data.x, data.edge_index)
        return self.decoder(h[mask])

@dataclass
class Config:
    mask_ratio:float=0.4; lr:float=1e-3; epochs:int=300; patience:int=40; ckpt:str="ckpt_gtr_graphmae.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

def node_mask(N, r, device):
    m = torch.zeros(N, dtype=torch.bool, device=device)
    m[torch.randperm(N, device=device)[:max(1,int(N*r))]] = True
    return m
