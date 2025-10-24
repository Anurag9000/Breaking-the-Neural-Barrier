import torch, torch.nn as nn, torch.nn.functional as F
from core.diffusion_core import SinusoidalTimeEmbedding


class WindowAttention(nn.Module):
    def __init__(self, dim, heads=4, window=8):
        super().__init__()
        self.heads = heads
        self.window = window
        self.to_qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        b, n, d = x.shape
        h = self.heads
        wsize = self.window
        qkv = self.to_qkv(x).reshape(b, n, 3, h, d // h).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) / (d // h) ** 0.5
        attn = attn.softmax(-1)
        out = (attn @ v).transpose(1, 2).reshape(b, n, d)
        return self.proj(out)


class SwinBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp=4):
        super().__init__()
        self.attn = WindowAttention(dim, heads)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, dim * mlp),
            nn.GELU(),
            nn.Linear(dim * mlp, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


class SwinDiff(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=192, depth=4, patch=4, cond_dim=0):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, 4096, dim))
        self.time = SinusoidalTimeEmbedding(dim)
        self.cproj = nn.Linear(cond_dim, dim) if cond_dim > 0 else None
        self.blocks = nn.ModuleList([SwinBlock(dim) for _ in range(depth)])
        self.out = nn.Linear(dim, dim)
        self.unpatch = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, x, t, cond=None):
        x = self.patch(x)
        b, c, h, w = x.shape
        tok = x.flatten(2).transpose(1, 2)
        t_emb = self.time(t)
        if self.cproj is not None and cond is not None:
            t_emb = t_emb + self.cproj(cond)
        tok = tok + t_emb.unsqueeze(1) + self.pos[:, :tok.size(1)]
        for blk in self.blocks:
            tok = blk(tok)
        tok = self.out(tok)
        tok = tok.transpose(1, 2).reshape(b, c, h, w)
        return self.unpatch(tok)
