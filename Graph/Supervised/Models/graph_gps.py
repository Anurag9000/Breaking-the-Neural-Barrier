import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GINConv

class GPSBlock(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.5):
        super().__init__()
        self.local = GINConv(nn.Sequential(nn.Linear(dim, dim), nn.ReLU(), nn.Linear(dim, dim)))
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
        self.dropout=dropout
    def forward(self, x, edge_index):
        x = x + self.local(x, edge_index)
        h = x.unsqueeze(0)
        y,_ = self.attn(self.ln1(h), self.ln1(h), self.ln1(h), need_weights=False)
        x = x + F.dropout(y.squeeze(0), p=self.dropout, training=self.training)
        x = x + F.dropout(self.ff(self.ln2(x)), p=self.dropout, training=self.training)
        return x

class GPSNet(nn.Module):
    def __init__(self, in_dim, dim=128, out_dim=7, layers=3, heads=4, dropout=0.5):
        super().__init__()
        self.enc = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([GPSBlock(dim, heads, dropout) for _ in range(layers)])
        self.out = nn.Linear(dim, out_dim)
    def forward(self, x, edge_index):
        x = self.enc(x)
        for blk in self.blocks:
            x = blk(x, edge_index)
        return self.out(x)
