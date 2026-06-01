import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class XLNetPermLM(nn.Module):
    """Permutation Language Modeling (simplified): random token order for attention mask; predict each token from others.
    Single Transformer encoder with permutation masks; objective is to predict token ids given permuted context.
    """
    def __init__(self, vocab, dim=512, depth=6, heads=8, mlp_ratio=4.0, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = nn.Embedding(max_len, dim)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, int(dim*mlp_ratio), batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
        self.max_len = max_len

    def perm_mask(self, L):
        perm = torch.randperm(L)
        mask = torch.ones(L, L)
        for i,p in enumerate(perm):
            mask[p, perm[:i+1]] = 0  # can attend to tokens earlier in permutation (including self -> zero out later)
        mask.fill_diagonal_(1)  # prevent trivial identity; we'll exclude self loss via shift
        return mask.bool(), perm

    def forward(self, x):
        B,L = x.shape
        pos = torch.arange(L, device=x.device)
        h = self.emb(x) + self.pos(pos).unsqueeze(0).expand(B,-1,-1)
        m, perm = self.perm_mask(L)
        m = m.to(x.device)
        for lyr in self.layers:
            h = lyr(h, src_mask=m)
        h = self.norm(h)
        logits = self.head(h)
        # predict each token using others (no shift)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), x.view(-1))
        return loss
