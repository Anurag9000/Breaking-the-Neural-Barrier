import torch
import torch.nn as nn
import torch.nn.functional as F

class CausalBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Linear(int(dim*mlp_ratio), dim))
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B,N,D = x.shape
        mask = torch.triu(torch.ones(N,N, device=x.device), diagonal=1).bool()
        x = x + self.drop(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)[0])
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class GPTLM(nn.Module):
    def __init__(self, vocab, dim=512, depth=8, heads=8, mlp_ratio=4.0, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.blocks = nn.ModuleList([CausalBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
    def forward(self, x):
        B,L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0).expand(B,-1)
        h = self.emb(x) + self.pos(pos)
        for blk in self.blocks:
            h = blk(h)
        h = self.norm(h)
        logits = self.head(h[:, :-1])
        targets = x[:, 1:]
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1))
        return loss
