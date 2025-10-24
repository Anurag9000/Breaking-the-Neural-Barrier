import torch
import torch.nn as nn
import torch.nn.functional as F
from core.diffusion_core import SinusoidalTimeEmbedding


class ResBlock1D(nn.Module):
    def __init__(self, c, tdim, c_out=None):
        super().__init__()
        c_out = c_out or c
        self.c1 = nn.Conv1d(c, c_out, 3, padding=1)
        self.c2 = nn.Conv1d(c_out, c_out, 3, padding=1)
        self.n1 = nn.GroupNorm(8, c_out)
        self.n2 = nn.GroupNorm(8, c_out)
        self.t = nn.Sequential(
            nn.SiLU(),
            nn.Linear(tdim, c_out)
        )
        self.skip = nn.Conv1d(c, c_out, 1) if c_out != c else nn.Identity()

    def forward(self, x, t_emb):
        h = self.n1(self.c1(x)) + self.t(t_emb).unsqueeze(-1)
        h = F.silu(h)
        h = self.n2(self.c2(h))
        return F.silu(h + self.skip(x))


class UNet1dDiff(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64, tdim=256):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)

        # Input layer
        self.inp = nn.Conv1d(in_ch, base, 3, padding=1)

        # Downsampling path
        self.d1 = ResBlock1D(base, tdim, base)
        self.d2 = ResBlock1D(base, tdim, base * 2)
        self.d3 = ResBlock1D(base * 2, tdim, base * 4)
        self.mid = ResBlock1D(base * 4, tdim)

        # Upsampling path
        self.u3 = ResBlock1D(base * 4, tdim, base * 2)
        self.u2 = ResBlock1D(base * 2, tdim, base)
        self.u1 = ResBlock1D(base, tdim, base)

        # Output layer
        self.out = nn.Conv1d(base, out_ch, 3, padding=1)

    def forward(self, x, t, cond=None):
        t_emb = self.time(t)

        # Encoder
        x0 = self.inp(x)
        d1 = self.d1(x0, t_emb)
        x1 = F.avg_pool1d(d1, 2)

        d2 = self.d2(x1, t_emb)
        x2 = F.avg_pool1d(d2, 2)

        d3 = self.d3(x2, t_emb)
        m = self.mid(F.avg_pool1d(d3, 2), t_emb)

        # Decoder
        u3 = F.interpolate(m, scale_factor=2, mode='nearest')
        u3 = self.u3(u3 + d3, t_emb)

        u2 = F.interpolate(u3, scale_factor=2, mode='nearest')
        u2 = self.u2(u2 + d2, t_emb)

        u1 = F.interpolate(u2, scale_factor=2, mode='nearest')
        u1 = self.u1(u1 + d1, t_emb)

        return self.out(u1)
