import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class GraphiTLike(nn.Module):
    def __init__(self, in_dim, dim=128, out_dim=7, layers=2, heads=4, rw_steps=4, dropout=0.5):
        super().__init__()
        self.dropout=dropout; self.rw_steps=rw_steps
        self.input = nn.Linear(in_dim, dim)
        self.rw_proj = nn.Linear(rw_steps, dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                'ln2': nn.LayerNorm(dim),
                'ff': nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
            }) for _ in range(layers)
        ])
        self.out = nn.Linear(dim, out_dim)

    def rw_pe(self, edge_index, num_nodes, device):
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
        A = A + torch.eye(num_nodes, device=device)
        D = A.sum(dim=1).clamp(min=1)
        P = A / D.unsqueeze(1)
        feats = [torch.eye(num_nodes, device=device)]
        x = P.clone()
        for _ in range(1, self.rw_steps):
            feats.append(x)
            x = x @ P
        PE = torch.stack([f.diag() for f in feats], dim=1)  # [N, K]
        return PE

    def forward(self, x, edge_index):
        N=x.size(0); device=x.device
        pe = self.rw_pe(edge_index, N, device)
        h = self.input(x) + self.rw_proj(pe)
        h = F.dropout(h, p=self.dropout, training=self.training).unsqueeze(0)
        for blk in self.blocks:
            y = blk['ln1'](h); y,_ = blk['attn'](y,y,y, need_weights=False)
            h = h + F.dropout(y, p=self.dropout, training=self.training)
            y = blk['ff'](blk['ln2'](h))
            h = h + F.dropout(y, p=self.dropout, training=self.training)
        h = h.squeeze(0)
        return self.out(h)
