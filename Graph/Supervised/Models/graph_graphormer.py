import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import to_dense_adj

class GraphormerLite(nn.Module):
    def __init__(self, in_dim, dim=128, out_dim=7, layers=2, heads=4, dropout=0.5):
        super().__init__()
        self.dropout=dropout
        self.input = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                'ln2': nn.LayerNorm(dim),
                'ff': nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
            }) for _ in range(layers)
        ])
        # hop distance embeddings: self(0), neighbors(1), 2-hop(2), far(3)
        self.hop_bias = nn.Embedding(4, heads)
        self.out = nn.Linear(dim, out_dim)

    def build_bias(self, edge_index, num_nodes, device):
        A = to_dense_adj(edge_index, max_num_nodes=num_nodes).squeeze(0).to(device)
        I = torch.eye(num_nodes, device=device)
        A1 = (A>0).float(); A2 = ((A1@A1) > 0).float()
        hop = torch.full((num_nodes,num_nodes), 3, device=device, dtype=torch.long)
        hop[A1.bool()] = 1
        hop[A2.bool()] = torch.minimum(hop[A2.bool()], torch.ones_like(hop[A2.bool()])*2)
        hop[I.bool()] = 0
        return hop  # [N,N] in {0,1,2,3}

    def forward(self, x, edge_index):
        N=x.size(0); device=x.device
        h = self.input(x); h = F.dropout(h, p=self.dropout, training=self.training).unsqueeze(0)
        hop = self.build_bias(edge_index, N, device)
        # convert head-wise bias to attn_mask (additive): [H, N, N]
        head_bias = self.hop_bias(hop)  # [N,N,H]
        head_bias = head_bias.permute(2,0,1)  # [H,N,N]
        attn_mask = head_bias.sum(0) * 0.0  # placeholder to satisfy API (per-head not directly supported)
        # We approximate by averaging biases over heads and adding to attn_mask
        attn_mask = head_bias.mean(0)
        for blk in self.blocks:
            y = blk['ln1'](h)
            y, _ = blk['attn'](y, y, y, attn_mask=attn_mask, need_weights=False)
            h = h + F.dropout(y, p=self.dropout, training=self.training)
            y = blk['ff'](blk['ln2'](h))
            h = h + F.dropout(y, p=self.dropout, training=self.training)
        h = h.squeeze(0)
        return self.out(h)
