import math
from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- UNet, time/sigma embedding ----------
class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim=dim
    def forward(self, s):
        half=self.dim//2
        freqs=torch.exp(torch.arange(half, device=s.device)*-(math.log(10000.0)/(half-1)))
        args=s[:,None]*freqs[None,:]
        emb=torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim%2: emb=F.pad(emb,(0,1))
        return emb

class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, edim):
        super().__init__()
        self.c1=nn.Conv2d(c_in,c_out,3,padding=1); self.b1=nn.BatchNorm2d(c_out)
        self.c2=nn.Conv2d(c_out,c_out,3,padding=1); self.b2=nn.BatchNorm2d(c_out)
        self.act=nn.SiLU(); self.emb=nn.Sequential(nn.SiLU(), nn.Linear(edim,c_out))
        self.skip=nn.Conv2d(c_in,c_out,1) if c_in!=c_out else nn.Identity()
    def forward(self,x,e):
        h=self.act(self.b1(self.c1(x))); h=h+self.emb(e)[:, :, None, None]
        h=self.b2(self.c2(h)); return self.act(h + self.skip(x))

class Down(nn.Module):
    def __init__(self,c_in,c_out,edim):
        super().__init__(); self.b1=ResBlock(c_in,c_out,edim); self.b2=ResBlock(c_out,c_out,edim); self.pool=nn.AvgPool2d(2)
    def forward(self,x,e):
        x=self.b1(x,e); x=self.b2(x,e); d=self.pool(x); return x,d

class Up(nn.Module):
    def __init__(self,c_in,c_out,edim):
        super().__init__(); self.b1=ResBlock(c_in,c_out,edim); self.b2=ResBlock(c_out,c_out,edim)
    def forward(self,x,skip,e):
        x=F.interpolate(x,scale_factor=2,mode='nearest'); x=torch.cat([x,skip],dim=1)
        x=self.b1(x,e); x=self.b2(x,e); return x

class EDMUNet(nn.Module):
    def __init__(self, in_ch=3, base=64, ch_mult=(1,2,4), edim=256, out_ch=3):
        super().__init__()
        self.eproj=nn.Sequential(SinusoidalEmbedding(edim), nn.Linear(edim,edim*4), nn.SiLU(), nn.Linear(edim*4, edim))
        c1,c2,c3=[base*m for m in ch_mult]
        self.in_conv=nn.Conv2d(in_ch,c1,3,padding=1)
        self.d1=Down(c1,c1,edim); self.d2=Down(c1,c2,edim)
        self.m1=ResBlock(c2,c3,edim); self.m2=ResBlock(c3,c3,edim)
        self.u1=Up(c3+c2,c2,edim); self.u2=Up(c2+c1,c1,edim)
        self.out_bn=nn.BatchNorm2d(c1); self.out_act=nn.SiLU(); self.out=nn.Conv2d(c1,out_ch,3,padding=1)
    def forward(self,x,sigma_embed):
        e=self.eproj(sigma_embed)
        x0=self.in_conv(x); s1,d1=self.d1(x0,e); s2,d2=self.d2(d1,e)
        m=self.m1(d2,e); m=self.m2(m,e); u1=self.u1(m,s2,e); u2=self.u2(u1,s1,e)
        h=self.out_act(self.out_bn(u2)); return self.out(h)

# ---------- EDM core ----------
@dataclass
class EDMConfig:
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0                  # Karras rho for sigma schedule

class EDM(nn.Module):
    def __init__(self, model: nn.Module, cfg: EDMConfig):
        super().__init__(); self.model=model; self.cfg=cfg
    def karras_sigmas(self, n: int, device) -> torch.Tensor:
        rho=self.cfg.rho; s0=self.cfg.sigma_min**(1/rho); s1=self.cfg.sigma_max**(1/rho)
        ramp=torch.linspace(0,1,n, device=device)
        sigmas=(s0 + ramp*(s1-s0))**rho
        return sigmas
    def loss(self, x0: torch.Tensor):
        B=x0.size(0)
        # sample per-example sigma from log-uniform within [sigma_min,sigma_max]
        rnd = torch.rand(B, device=x0.device)
        sig = self.cfg.sigma_min * (self.cfg.sigma_max/self.cfg.sigma_min)**rnd
        sig_ = sig[:,None,None,None]
        noise=torch.randn_like(x0)
        x_noisy = x0 + sig_*noise
        # predict noise (eps) with sigma conditioning
        pred = self.model(x_noisy, sig)
        # EDM weighting (predicting noise): weight ~ sig^2
        w = (sig**2)
        return (w*((pred - noise)**2).flatten(1).mean(dim=1)).mean()
    @torch.no_grad()
    def sample(self, shape, device, steps=40):
        x=torch.randn(shape, device=device)*self.cfg.sigma_max
        sigmas=self.karras_sigmas(steps, device)
        for i in range(steps-1):
            sigma=sigmas[i]; sigma_next=sigmas[i+1]
            s=torch.full((shape[0],), float(sigma), device=device)
            v=self.model(x, s)  # predict noise
            # deterministic step (Heun-like second-order)
            dt = sigma_next - sigma
            d = -v
            x_euler = x + d*dt
            # second eval
            s_next=torch.full((shape[0],), float(sigma_next), device=device)
            v2=self.model(x_euler, s_next)
            d2 = -v2
            x = x + 0.5*(d + d2)*dt
        return x.clamp(-1,1)

def count_parameters(m):
    return sum(p.numel() for p in m.parameters())
