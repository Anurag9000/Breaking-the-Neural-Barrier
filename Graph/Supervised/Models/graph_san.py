import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.utils import get_laplacian

class SANEncoder(nn.Module):
    def __init__(self, in_dim, dim=128, out_dim=7, layers=2, heads=4, pe_dim=16, dropout=0.5):
        super().__init__()
        self.dropout=dropout
        self.input = nn.Linear(in_dim, dim)
        self.pe_proj = nn.Linear(pe_dim, dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'attn': nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                'ln2': nn.LayerNorm(dim),
                'ff': nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
            }) for _ in range(layers)
        ])
        self.out = nn.Linear(dim, out_dim)

    def laplacian_pe(self, edge_index, num_nodes, k=16, device='cpu'):
        # compute top-k Laplacian eigenvectors (smallest non-zero)
        edge_index = torch.cat([edge_index, torch.stack([torch.arange(num_nodes, device=device), torch.arange(num_nodes, device=device)])], dim=1)
        edge_weight = torch.ones(edge_index.size(1), device=device)
        L_index, L_weight = get_laplacian(edge_index, edge_weight, normalization='sym', num_nodes=num_nodes)
        L = torch.sparse_coo_tensor(L_index, L_weight, (num_nodes, num_nodes)).to_dense()
        evals, evecs = torch.linalg.eigh(L)
        k = min(k, num_nodes)
        return evecs[:, :k]

    def forward(self, x, edge_index):
        N = x.size(0); device=x.device
        pe = self.laplacian_pe(edge_index, N, k=self.pe_proj.in_features, device=device)
        h = self.input(x) + self.pe_proj(pe)
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h.unsqueeze(0)  # [1, N, D]
        for blk in self.blocks:
            y = blk['ln1'](h)
            y, _ = blk['attn'](y, y, y, need_weights=False)
            h = h + F.dropout(y, p=self.dropout, training=self.training)
            y = blk['ff'](blk['ln2'](h))
            h = h + F.dropout(y, p=self.dropout, training=self.training)
        h = h.squeeze(0)
        return self.out(h)
