import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class EGTLike(nn.Module):
    """Edge-augmented Graph Transformer (lite): nodes attend globally with edge-aware bias.
    We build an additive bias from adjacency and optional edge attributes.
    """
    def __init__(self, in_dim, dim=128, out_dim=7, layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.dropout=dropout; self.heads=heads
        self.input = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                'ln2': nn.LayerNorm(dim),
                'ff': nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
            }) for _ in range(layers)
        ])
        self.edge_bias_lin = nn.Linear(1, heads)  # scalar edge weight -> per-head bias
        self.out = nn.Linear(dim, out_dim)

    def bias_from_edges(self, edge_index, edge_attr, num_nodes, device):
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
        if edge_attr is not None and edge_attr.dim()==2 and edge_attr.size(1)>=1:
            w = torch.zeros_like(A)
            w[edge_index[0], edge_index[1]] = edge_attr[:,0]
        else:
            w = A
        b = self.edge_bias_lin(w.unsqueeze(-1))  # [N,N,H]
        return b.permute(2,0,1)  # [H,N,N]

    def forward(self, x, edge_index, edge_attr=None):
        N=x.size(0); device=x.device
        h = self.input(x); h = F.dropout(h, p=self.dropout, training=self.training).unsqueeze(0)
        bias = self.bias_from_edges(edge_index, edge_attr, N, device)
        attn_mask = bias.mean(0)  # approximate per-head bias
        for blk in self.blocks:
            y = blk['ln1'](h)
            y, _ = blk['attn'](y,y,y, attn_mask=attn_mask, need_weights=False)
            h = h + F.dropout(y, p=self.dropout, training=self.training)
            y = blk['ff'](blk['ln2'](h))
            h = h + F.dropout(y, p=self.dropout, training=self.training)
        return self.out(h.squeeze(0))
