import torch
import torch.nn as nn
import torch.nn.functional as F
from core.diffusion_core import SinusoidalTimeEmbedding


class DWConv(nn.Module):
    def __init__(self, c_in, c_out, stride=1):
        super().__init__()
        self.dw = nn.Conv2d(c_in, c_in, 3, padding=1, stride=stride, groups=c_in)
        self.pw = nn.Conv2d(c_in, c_out, 1)
        self.bn = nn.BatchNorm2d(c_out)

    def forward(self, x):
        return F.silu(self.bn(self.pw(self.dw(x))))


class TinyMobileUNet(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=32, tdim=128, cond_dim=0):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)
        self.cproj = nn.Linear(cond_dim, tdim) if cond_dim > 0 else None

        self.inp = DWConv(in_ch, base)
        self.d1 = DWConv(base, base)
        self.d2 = DWConv(base, base*2)
        self.d3 = DWConv(base*2, base*4)
        self.mid = DWConv(base*4, base*4)
        self.u3 = DWConv(base*4, base*2)
        self.u2 = DWConv(base*2, base)
        self.u1 = DWConv(base, base)
        self.out = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x, t, cond=None):
        t_emb = self.time(t)
        if self.cproj is not None and cond is not None:
            t_emb = t_emb + self.cproj(cond)

        x0 = self.inp(x)
        d1 = self.d1(x0); x1 = F.avg_pool2d(d1, 2)
        d2 = self.d2(x1); x2 = F.avg_pool2d(d2, 2)
        d3 = self.d3(x2); m = self.mid(F.avg_pool2d(d3, 2))
        
        u3 = F.interpolate(m, scale_factor=2, mode='nearest'); u3 = self.u3(u3 + d3)
        u2 = F.interpolate(u3, scale_factor=2, mode='nearest'); u2 = self.u2(u2 + d2)
        u1 = F.interpolate(u2, scale_factor=2, mode='nearest'); u1 = self.u1(u1 + d1)

        return self.out(u1)
