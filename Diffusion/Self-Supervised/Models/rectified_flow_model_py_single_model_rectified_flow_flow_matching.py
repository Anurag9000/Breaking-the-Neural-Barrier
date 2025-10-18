import math
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

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

# ---------- UNet predicting velocity field v(x_t,t) ----------
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

class RFUNet(nn.Module):
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

# ---------- Rectified Flow objective ----------
@dataclass
class RFConfig:
    continuous: bool = True

class RectifiedFlow(nn.Module):
    def __init__(self, model: nn.Module):
        super().__init__(); self.model=model
    def loss(self, x0: torch.Tensor):
        B=x0.size(0); device=x0.device
        t=torch.rand(B, device=device)
        z=torch.randn_like(x0)
        x_t = (1 - t)[:,None,None,None]*x0 + t[:,None,None,None]*z
        v_true = z - x0
        v_pred = self.model(x_t, t)
        return F.mse_loss(v_pred, v_true)
    @torch.no_grad()
    def sample(self, shape, device, steps=50):
        z=torch.randn(shape, device=device)
        x=z
        for i in reversed(range(steps)):
            t=torch.full((shape[0],), (i+1)/steps, device=device)
            v=self.model(x,t)
            dt=1.0/steps
            x = x - v*dt  # integrate backward
        return x.clamp(-1,1)

def count_parameters(m):
    return sum(p.numel() for p in m.parameters())
