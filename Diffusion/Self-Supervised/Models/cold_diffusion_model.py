import math
from dataclasses import dataclass
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import gaussian_blur

# ---------- time embedding ----------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__(); self.dim=dim
    def forward(self,t):
        half=self.dim//2
        freqs=torch.exp(torch.arange(half, device=t.device)*-(math.log(10000.0)/(half-1)))
        args=t[:,None]*freqs[None,:]
        emb=torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim%2: emb=F.pad(emb,(0,1))
        return emb

# ---------- UNet ----------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, tdim):
        super().__init__()
        self.c1=nn.Conv2d(c_in,c_out,3,padding=1); self.b1=nn.BatchNorm2d(c_out)
        self.c2=nn.Conv2d(c_out,c_out,3,padding=1); self.b2=nn.BatchNorm2d(c_out)
        self.emb=nn.Sequential(nn.SiLU(), nn.Linear(tdim, c_out))
        self.act=nn.SiLU(); self.skip=nn.Conv2d(c_in,c_out,1) if c_in!=c_out else nn.Identity()
    def forward(self,x,temb):
        h=self.act(self.b1(self.c1(x))); h=h+self.emb(temb)[:, :, None, None]
        h=self.b2(self.c2(h)); return self.act(h + self.skip(x))

class Down(nn.Module):
    def __init__(self,c_in,c_out,tdim):
        super().__init__(); self.b1=ResBlock(c_in,c_out,tdim); self.b2=ResBlock(c_out,c_out,tdim); self.pool=nn.AvgPool2d(2)
    def forward(self,x,t):
        x=self.b1(x,t); x=self.b2(x,t); d=self.pool(x); return x,d

class Up(nn.Module):
    def __init__(self,c_in,c_out,tdim):
        super().__init__(); self.b1=ResBlock(c_in,c_out,tdim); self.b2=ResBlock(c_out,c_out,tdim)
    def forward(self,x,skip,t):
        x=F.interpolate(x,scale_factor=2,mode='nearest'); x=torch.cat([x,skip],dim=1)
        x=self.b1(x,t); x=self.b2(x,t); return x

class ColdUNet(nn.Module):
    def __init__(self, in_ch=3, base=64, ch_mult=(1,2,4), tdim=256, out_ch=3):
        super().__init__()
        self.tproj=nn.Sequential(SinusoidalTimeEmbedding(tdim), nn.Linear(tdim, tdim*4), nn.SiLU(), nn.Linear(tdim*4, tdim))
        c1,c2,c3=[base*m for m in ch_mult]
        self.in_conv=nn.Conv2d(in_ch,c1,3,padding=1)
        self.d1=Down(c1,c1,tdim); self.d2=Down(c1,c2,tdim)
        self.m1=ResBlock(c2,c3,tdim); self.m2=ResBlock(c3,c3,tdim)
        self.u1=Up(c3+c2,c2,tdim); self.u2=Up(c2+c1,c1,tdim)
        self.out_bn=nn.BatchNorm2d(c1); self.out_act=nn.SiLU(); self.out=nn.Conv2d(c1,out_ch,3,padding=1)
    def forward(self,x,t):
        t=self.tproj(t); x0=self.in_conv(x); s1,d1=self.d1(x0,t); s2,d2=self.d2(d1,t)
        m=self.m1(d2,t); m=self.m2(m,t); u1=self.u1(m,s2,t); u2=self.u2(u1,s1,t)
        h=self.out_act(self.out_bn(u2)); return self.out(h)

# ---------- Deterministic degradation operators ----------
@dataclass
class ColdCfg:
    T:int=1000
    mode: Literal['blur','mask'] = 'blur'

class ColdDiffusion(nn.Module):
    def __init__(self, model: nn.Module, cfg: ColdCfg):
        super().__init__(); self.model=model; self.cfg=cfg
    def degrade(self, x: torch.Tensor, t: torch.Tensor):
        # t in [0, T-1]; map to strength s in [0,1]
        s = t.float() / max(self.cfg.T-1, 1)
        if self.cfg.mode == 'blur':
            # kernel size odd, sigma ~ 0.5 + 5*s
            k = 3 + (s*5).long()*2
            # vectorized per-sample blur via loop (simple and clear)
            xs=[]
            for i in range(x.size(0)):
                ki=int(k[i].item()); ki = max(3, ki | 1)
                xs.append(gaussian_blur(x[i], [ki,ki], [0.5 + 5*s[i].item(), 0.5 + 5*s[i].item()]))
            x_deg = torch.stack(xs, dim=0)
        else:  # 'mask'
            # random box mask area proportional to s
            x_deg=x.clone()
            B,C,H,W=x.shape
            for i in range(B):
                frac=min(0.8, 0.05 + 0.7*s[i].item())
                h=int(H*math.sqrt(frac)); w=int(W*math.sqrt(frac))
                y=torch.randint(0, max(1,H-h+1), (1,)).item(); x0=torch.randint(0, max(1,W-w+1), (1,)).item()
                x_deg[i,:, y:y+h, x0:x0+w] = 0.0
        return x_deg
    def loss(self, x0: torch.Tensor):
        B=x0.size(0); device=x0.device
        t=torch.randint(0, self.cfg.T, (B,), device=device)
        x_t = self.degrade(x0, t)
        x0_pred = self.model(x_t, t.float())
        return F.mse_loss(x0_pred, x0)
    @torch.no_grad()
    def inverse(self, x_deg: torch.Tensor, steps=20):
        # fixed-point refinement by applying model multiple times
        x=x_deg
        for _ in range(steps):
            t=torch.zeros(x.size(0), device=x.device)  # treat as weakest corruption
            x = self.model(x, t)
        return x.clamp(-1,1)

def count_parameters(m):
    return sum(p.numel() for p in m.parameters())
