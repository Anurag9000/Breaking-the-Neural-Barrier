import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class ExphormerLite(nn.Module):
    """Sparse global attention via adjacency + random expander edges (approximation)."""
    def __init__(self, in_dim, dim=128, out_dim=7, layers=2, heads=4, dropout=0.5, extra_global=16):
        super().__init__()
        self.dropout=dropout; self.heads=heads; self.extra_global=extra_global
        self.input = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                'ln2': nn.LayerNorm(dim),
                'ff': nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
            }) for _ in range(layers)
        ])
        self.out = nn.Linear(dim, out_dim)
    def make_mask(self, edge_index, N, device):
        A = to_dense_adj(edge_index, max_num_nodes=N).squeeze(0).to(device)
        # add random expander edges
        if self.extra_global>0:
            idx = torch.randint(0, N, (self.extra_global,2), device=device)
            A[idx[:,0], idx[:,1]] = 1
        mask = (A==0).float() * (-1e4)  # disallow non-edges
        return mask
    def forward(self, x, edge_index):
        N=x.size(0); device=x.device
        h = self.input(x); h = F.dropout(h, p=self.dropout, training=self.training).unsqueeze(0)
        attn_mask = self.make_mask(edge_index, N, device)
        for blk in self.blocks:
            y = blk['ln1'](h)
            y,_ = blk['attn'](y,y,y, attn_mask=attn_mask, need_weights=False)
            h = h + F.dropout(y, p=self.dropout, training=self.training)
            y = blk['ff'](blk['ln2'](h))
            h = h + F.dropout(y, p=self.dropout, training=self.training)
        return self.out(h.squeeze(0))
