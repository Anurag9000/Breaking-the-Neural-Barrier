import torch
import torch.nn as nn

# ----------------------------
# Linformer: project keys/values along sequence dimension to low rank k
# ----------------------------
class LinformerSelfAttn(nn.Module):
    def __init__(self, dim, heads=8, k=64):
        super().__init__(); self.h=heads; self.dim=dim; self.k=k; self.dk=dim//heads
        self.q = nn.Linear(dim, dim); self.kv = nn.Linear(dim, 2*dim); self.proj_k = nn.Linear(0,0, bias=False)
        self.E = nn.Parameter(torch.randn(1, k, 1024))  # supports up to 1024 tokens; will slice
        self.proj = nn.Linear(dim, dim)
        self.ln1 = nn.LayerNorm(dim); self.ln2 = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(nn.Linear(dim, 4*dim), nn.GELU(), nn.Linear(4*dim, dim))
    def forward(self, x):
        B,S,D = x.shape; H=self.h; dh=self.dk
        x1 = self.ln1(x)
        q = self.q(x1).view(B,S,H,dh)
        kv = self.kv(x1).view(B,S,2,H,dh)
        k = kv[:,:,0]; v = kv[:,:,1]
        # project sequence S->k via learned E (shared across heads for simplicity)
        E = self.E[:, :self.k, :S]  # (1,k,S)
        k_lin = torch.einsum('bshd,bks->bkhd', k, E)
        v_lin = torch.einsum('bshd,bks->bkhd', v, E)
        attn = (q.unsqueeze(-2) @ k_lin.transpose(-1,-2)) / (dh**0.5)  # B,S,H,1,k
        w = attn.softmax(-1)
        out = (w @ v_lin.unsqueeze(-3)).squeeze(-3)  # B,S,H,dh
        out = out.view(B,S,D)
        x = x + self.proj(out)
        x = x + self.ffn(self.ln2(x))
        return x

class LinformerEncoder(nn.Module):
    def __init__(self, vocab, num_classes, dim=256, depth=6, heads=8, k=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([LinformerSelfAttn(dim, heads, k) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim); self.head = nn.Linear(dim, num_classes)
    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x[:,0]))
