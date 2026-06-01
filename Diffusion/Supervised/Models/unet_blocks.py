import torch
import torch.nn as nn
import torch.nn.functional as F
from .diffusion_core import SinusoidalTimeEmbedding


# -----------------------------
# Residual Block
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, c, tdim, c_out=None):
        super().__init__()
        self.c_out = c_out or c
        self.conv1 = nn.Conv2d(c, self.c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(self.c_out, self.c_out, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(tdim, self.c_out))
        self.skip = nn.Conv2d(c, self.c_out, 1) if self.c_out != c else nn.Identity()
        self.norm1 = nn.GroupNorm(8, self.c_out)
        self.norm2 = nn.GroupNorm(8, self.c_out)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.norm1(h) + self.time(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.norm2(h)
        return F.silu(h + self.skip(x))


# -----------------------------
# Simple U-Net for Diffusion
# -----------------------------
class SimpleUNet(nn.Module):
    def __init__(self, in_ch, out_ch, base=64, tdim=256, cond_dim=0):
        super().__init__()
        chs = [base, base * 2, base * 4]

        # Time embedding network
        self.time_mlp = nn.Sequential(
            SinusoidalTimeEmbedding(tdim),
            nn.SiLU()
        )

        # Optional conditioning (e.g., class embedding)
        self.cond = nn.Linear(cond_dim, tdim) if cond_dim > 0 else None

        # Downsampling path
        self.inp = nn.Conv2d(in_ch, base, 3, padding=1)
        self.down1 = ResBlock(base, tdim, chs[0])
        self.down2 = ResBlock(chs[0], tdim, chs[1])
        self.down3 = ResBlock(chs[1], tdim, chs[2])

        # Bottleneck
        self.mid = ResBlock(chs[2], tdim)

        # Upsampling path
        self.up3 = ResBlock(chs[2], tdim, chs[1])
        self.up2 = ResBlock(chs[1], tdim, chs[0])
        self.up1 = ResBlock(chs[0], tdim, base)

        # Output convolution
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x, t, cond=None):
        # Embed time
        t_emb = self.time_mlp(t)

        # Add conditioning if provided
        if self.cond is not None and cond is not None:
            t_emb = t_emb + self.cond(cond)

        # Down path
        x0 = self.inp(x)
        d1 = self.down1(x0, t_emb)
        x1 = F.avg_pool2d(d1, 2)
        d2 = self.down2(x1, t_emb)
        x2 = F.avg_pool2d(d2, 2)
        d3 = self.down3(x2, t_emb)
        m = self.mid(F.avg_pool2d(d3, 2), t_emb)

        # Up path
        u3 = F.interpolate(m, scale_factor=2, mode='nearest')
        u3 = self.up3(u3 + d3, t_emb)

        u2 = F.interpolate(u3, scale_factor=2, mode='nearest')
        u2 = self.up2(u2 + d2, t_emb)

        u1 = F.interpolate(u2, scale_factor=2, mode='nearest')
        u1 = self.up1(u1 + d1, t_emb)

        return self.out(u1)
