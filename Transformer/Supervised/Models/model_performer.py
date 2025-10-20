import torch
import torch.nn as nn

# ----------------------------
# Minimal Performer encoder with FAVOR+ random features (elu version)
# ----------------------------
class LinearAttention(nn.Module):
    def __init__(self, dim, heads=8, nb_features=128):
        super().__init__()
        self.heads=heads; self.dim=dim; self.nb=nb_features; self.dk=dim//heads
        self.q = nn.Linear(dim, dim); self.k = nn.Linear(dim, dim); self.v = nn.Linear(dim, dim)
        self.proj = nn.Linear(dim, dim)
        self.register_buffer('omega', torch.randn(self.nb, self.dk))
    def _phi(self, x):
        return torch.elu(x) + 1
    def forward(self, x):
        B,S,D = x.shape; H=self.heads; dh=self.dk
        q = self.q(x).view(B,S,H,dh); k = self.k(x).view(B,S,H,dh); v = self.v(x).view(B,S,H,dh)
        qf = self._phi(q); kf = self._phi(k)
        kv = (kf.unsqueeze(-1) * v.unsqueeze(-2)).sum(dim=1)  # B,H,dh,dh
        z = 1.0 / (qf.sum(dim=1, keepdim=True) + 1e-6)
        out = (qf @ kv) * z
        out = out.view(B,S,D)
        return self.proj(out)

class PerformerEncoder(nn.Module):
    def __init__(self, vocab, num_classes, dim=256, depth=6, heads=8):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([])
        for _ in range(depth):
            self.blocks.append(nn.Sequential(
                nn.LayerNorm(dim), LinearAttention(dim, heads),
                nn.LayerNorm(dim), nn.Sequential(nn.Linear(dim, dim*4), nn.GELU(), nn.Linear(dim*4, dim))
            ))
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    def forward(self, ids):
        x = self.emb(ids)
        for i in range(0, len(self.blocks)):
            ln1, attn, ln2, ffn = self.blocks[i]
            x = x + attn(ln1(x))
            x = x + ffn(ln2(x))
        return self.head(self.norm(x[:,0]))
