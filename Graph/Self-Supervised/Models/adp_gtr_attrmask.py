import torch, torch.nn as nn, torch.nn.functional as F, random
from dataclasses import dataclass
from torch_geometric.nn import TransformerConv

class GTR(nn.Module):
    def __init__(self, in_dim, hid=128, layers=3, heads=4, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([TransformerConv(in_dim if i==0 else hid, hid//heads, heads=heads, dropout=dropout) for i in range(layers)])
        self.dec = nn.Linear(hid, in_dim)   # reconstruct masked features
        self.dropout = dropout
    def encode(self, x, ei):
        for conv in self.layers:
            x = F.relu(conv(x, ei)); x = F.dropout(x, p=self.dropout, training=self.training)
        return x
    def forward(self, data, mask):
        h = self.encode(data.x, data.edge_index)
        return self.dec(h[mask])

@dataclass
class Config:
    mask_ratio:float=0.3; lr:float=1e-3; epochs:int=300; patience:int=30; ckpt:str="ckpt_gtr_attrmask.pt"
    device:str="cuda" if torch.cuda.is_available() else "cpu"

def make_mask(N, ratio, device):
    idx = torch.randperm(N, device=device)
    k = max(1, int(N*ratio))
    mask = torch.zeros(N, dtype=torch.bool, device=device); mask[idx[:k]] = True
    return mask

class AttrMask_GTR(nn.Module):
    def __init__(self, in_dim, cfg:Config):
        super().__init__()
        self.backbone = GTR(in_dim); self.cfg = cfg
    def loss(self, data):
        N = data.x.size(0); mask = make_mask(N, self.cfg.mask_ratio, data.x.device)
        pred = self.backbone(data, mask)
        target = data.x[mask]
        return F.mse_loss(pred, target)
