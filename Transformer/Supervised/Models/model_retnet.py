import torch
import torch.nn as nn

class Retention(nn.Module):
    """Simplified RetNet retention block using lower-triangular causal kernel (no recurrence for simplicity)."""
    def __init__(self, d_model, heads=4):
        super().__init__()
        self.heads=heads; self.dk=d_model//heads
        self.q = nn.Linear(d_model, d_model); self.k = nn.Linear(d_model, d_model); self.v = nn.Linear(d_model, d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
    def forward(self, x):
        B,S,D = x.shape; H=self.heads; dh=self.dk
        x1 = self.ln1(x)
        q = self.q(x1).view(B,S,H,dh)
        k = self.k(x1).view(B,S,H,dh)
        v = self.v(x1).view(B,S,H,dh)
        # causal retention via masked attention-like op
        scores = (q.unsqueeze(3) * k.unsqueeze(2)).sum(-1) / (dh**0.5)  # B,S,H,S
        mask = torch.triu(torch.ones(S,S, device=x.device), diagonal=1).bool()
        scores = scores.permute(0,2,1,3)
        scores[..., mask] = -1e9
        attn = torch.softmax(scores, dim=-1)
        y = (attn @ v.transpose(1,2))  # B,H,S,dh
        y = y.permute(0,2,1,3).contiguous().view(B,S,D)
        x = x + self.proj(y)
        x = x + self.mlp(self.ln2(x))
        return x

class RetNetClassifier(nn.Module):
    def __init__(self, vocab, num_classes, d_model=256, depth=6, heads=4, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.blocks = nn.ModuleList([Retention(d_model, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x[:,0]))
