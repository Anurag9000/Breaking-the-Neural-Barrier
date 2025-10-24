import torch
import torch.nn as nn
import torch.nn.functional as F
from core.diffusion_core import SinusoidalTimeEmbedding


class CNXBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dw = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)  # depthwise conv
        self.norm = nn.LayerNorm(dim)
        self.pw1 = nn.Linear(dim, dim * 4)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(dim * 4, dim)

    def forward(self, x):
        h = self.dw(x)                          # [B, C, H, W]
        h = h.permute(0, 2, 3, 1)              # [B, H, W, C] for LayerNorm
        h = self.pw2(self.act(self.pw1(self.norm(h))))
        return h.permute(0, 3, 1, 2)           # back to [B, C, H, W]


class ConvNeXtDiff(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=96, stages=(3, 3, 9, 3), cond_dim=0):
        super().__init__()
        self.stem = nn.Conv2d(in_ch, dim, 4, stride=4)  # downsample
        self.time = SinusoidalTimeEmbedding(dim)
        self.cproj = nn.Linear(cond_dim, dim) if cond_dim > 0 else None
        self.stages = nn.ModuleList([nn.Sequential(*[CNXBlock(dim) for _ in range(n)]) for n in stages])
        self.head = nn.Sequential(
            nn.ConvTranspose2d(dim, dim, 4, stride=4),   # upsample
            nn.Conv2d(dim, out_ch, 3, padding=1)
        )

    def forward(self, x, t, cond=None):
        h = self.stem(x)
        t_emb = self.time(t)
        if self.cproj is not None and cond is not None:
            t_emb = t_emb + self.cproj(cond)
        h = h + t_emb.unsqueeze(-1).unsqueeze(-1)
        for stage in self.stages:
            h = stage(h)
        return self.head(h)
