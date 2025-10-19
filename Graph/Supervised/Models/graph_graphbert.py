import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphBERTLike(nn.Module):
    def __init__(self, in_dim, dim=128, out_dim=7, layers=2, heads=4, max_nodes=10000, dropout=0.5):
        super().__init__()
        self.dropout=dropout
        self.input = nn.Linear(in_dim, dim)
        self.pos_embed = nn.Embedding(max_nodes, dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                'ln2': nn.LayerNorm(dim),
                'ff': nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
            }) for _ in range(layers)
        ])
        self.out = nn.Linear(dim, out_dim)

    def forward(self, x, edge_index):
        N = x.size(0)
        h = self.input(x) + self.pos_embed.weight[:N]
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h.unsqueeze(0)
        for blk in self.blocks:
            y = blk['ln1'](h)
            y, _ = blk['attn'](y, y, y, need_weights=False)
            h = h + F.dropout(y, p=self.dropout, training=self.training)
            y = blk['ff'](blk['ln2'](h))
            h = h + F.dropout(y, p=self.dropout, training=self.training)
        h = h.squeeze(0)
        return self.out(h)
