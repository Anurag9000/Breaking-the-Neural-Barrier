import torch
import torch.nn as nn
import torch.nn.functional as F

class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dropout=0.5):
        super().__init__(); self.dim=dim; self.heads=heads; self.dropout=dropout
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
    def feature(self, x):
        return F.elu(x)+1  # positive feature map
    def forward(self, x):
        B=1; N=x.size(0); H=self.heads; D=self.dim//H
        q=self.q(x).view(B,N,H,D); k=self.k(x).view(B,N,H,D); v=self.v(x).view(B,N,H,D)
        q=self.feature(q); k=self.feature(k)
        # (B,N,H,D) -> (B,H,D,N) for k
        k_t = k.permute(0,2,3,1)
        kv = torch.matmul(k_t, v.permute(0,2,1,3))  # (B,H,D,D)
        z = 1.0/(torch.matmul(q, k_t.sum(-1).unsqueeze(-1))+1e-6)
        out = torch.matmul(q, kv)
        out = out * z
        out = out.permute(0,2,1,3).contiguous().view(N, H*D)
        return self.proj(out)

class LinearGraphTransformer(nn.Module):
    def __init__(self, in_dim, dim=128, out_dim=7, layers=2, heads=4, dropout=0.5):
        super().__init__(); self.dropout=dropout
        self.enc = nn.Linear(in_dim, dim)
        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1': nn.LayerNorm(dim),
                'attn': LinearAttention(dim, heads, dropout),
                'ln2': nn.LayerNorm(dim),
                'ff': nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(4*dim, dim))
            }) for _ in range(layers)
        ])
        self.out = nn.Linear(dim, out_dim)
    def forward(self, x, edge_index):
        h = self.enc(x)
        for blk in self.blocks:
            y = blk['attn'](blk['ln1'](h))
            h = h + F.dropout(y, p=self.dropout, training=self.training)
            y = blk['ff'](blk['ln2'](h))
            h = h + F.dropout(y, p=self.dropout, training=self.training)
        return self.out(h)
