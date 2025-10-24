import torch
import torch.nn as nn
import torch.nn.functional as F
from core.diffusion_core import SinusoidalTimeEmbedding


class ConvBNAct(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU()
        )

    def forward(self, x):
        return self.c(x)


class UNetPP(nn.Module):
    def __init__(self, in_ch, out_ch, base=64, tdim=256, cond_dim=0):
        super().__init__()
        self.time = nn.Sequential(SinusoidalTimeEmbedding(tdim), nn.SiLU())
        self.cproj = nn.Linear(cond_dim, tdim) if cond_dim > 0 else None

        c = base
        # Encoder
        self.enc1 = ConvBNAct(in_ch, c)
        self.enc2 = ConvBNAct(c, c*2)
        self.enc3 = ConvBNAct(c*2, c*4)
        self.mid = ConvBNAct(c*4, c*8)

        # Nested decoder
        self.up31 = ConvBNAct(c*8 + c*4, c*4)
        self.up21 = ConvBNAct(c*4 + c*2, c*2)
        self.up11 = ConvBNAct(c*2 + c, c)

        # Output
        self.out = nn.Conv2d(c, out_ch, 3, padding=1)

    def forward(self, x, t, cond=None):
        # Time embedding
        t_emb = self.time[0](t)
        t_emb = self.time[1](t_emb)
        if self.cproj is not None and cond is not None:
            t_emb = t_emb + self.cproj(cond)

        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        m = self.mid(F.avg_pool2d(e3, 2))

        # Decoder with skip connections
        u3 = F.interpolate(m, scale_factor=2, mode='nearest')
        u3 = self.up31(torch.cat([u3, e3], dim=1))

        u2 = F.interpolate(u3, scale_factor=2, mode='nearest')
        u2 = self.up21(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(u2, scale_factor=2, mode='nearest')
        u1 = self.up11(torch.cat([u1, e1], dim=1))

        return self.out(u1)
