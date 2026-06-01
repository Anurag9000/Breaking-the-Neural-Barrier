import torch
import torch.nn as nn

# ----------------------------
# Simplified Reformer-style encoder (bucketed/local attention in lieu of full LSH for brevity)
# ----------------------------
class LocalLSHAttention(nn.Module):
    def __init__(self, dim, heads=8, bucket_size=64):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.bucket = bucket_size
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
    def forward(self, x):
        B,S,D = x.shape; bs = self.bucket
        x1 = self.ln1(x)
        # split into buckets along sequence
        chunks = []
        for i in range(0, S, bs):
            seg = x1[:, i:i+bs, :]
            y = self.attn(seg, seg, seg, need_weights=False)[0]
            chunks.append(y)
        y = torch.cat(chunks, dim=1)
        x = x + y
        x = x + self.mlp(self.ln2(x))
        return x

class ReformerEncoder(nn.Module):
    def __init__(self, vocab, num_classes, dim=256, depth=6, heads=8, bucket=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([LocalLSHAttention(dim, heads, bucket) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x[:,0]))
