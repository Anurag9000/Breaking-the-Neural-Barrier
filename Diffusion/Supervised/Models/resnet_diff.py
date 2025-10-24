import torch
import torch.nn as nn
import torch.nn.functional as F
from core.diffusion_core import SinusoidalTimeEmbedding


class BasicBlock(nn.Module):
    def __init__(self, c, tdim):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1)
        self.c2 = nn.Conv2d(c, c, 3, padding=1)
        self.n1 = nn.GroupNorm(8, c)
        self.n2 = nn.GroupNorm(8, c)
        self.t = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, c)
        )

    def forward(self, x, t_emb):
        h = self.n1(self.c1(x)) + self.t(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.n2(self.c2(h))
        return F.silu(h + x)


class ResNetDiff(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, layers=(2, 2, 2, 2), tdim=256, cond_dim=0):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)
        self.cproj = nn.Linear(cond_dim, tdim) if cond_dim > 0 else None
        self.stem = nn.Conv2d(in_ch, base, 3, padding=1)
        self.layer1 = nn.ModuleList([BasicBlock(base, tdim) for _ in range(layers[0])])
        self.layer2 = nn.ModuleList([BasicBlock(base, tdim) for _ in range(layers[1])])
        self.layer3 = nn.ModuleList([BasicBlock(base, tdim) for _ in range(layers[2])])
        self.layer4 = nn.ModuleList([BasicBlock(base, tdim) for _ in range(layers[3])])
        self.head = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x, t, cond=None):
        t_emb = self.time(t)
        if self.cproj is not None and cond is not None:
            t_emb = t_emb + self.cproj(cond)

        h = self.stem(x)

        for blk in self.layer1:
            h = blk(h, t_emb)
        h = F.avg_pool2d(h, 2)

        for blk in self.layer2:
            h = blk(h, t_emb)
        h = F.interpolate(h, scale_factor=2, mode='nearest')

        for blk in self.layer3:
            h = blk(h, t_emb)

        for blk in self.layer4:
            h = blk(h, t_emb)

        return self.head(h)
