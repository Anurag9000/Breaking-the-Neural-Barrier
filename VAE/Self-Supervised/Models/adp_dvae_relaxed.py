from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    n_latents: int = 64
    width: int = 128
    depth: int = 2
    tau: float = 0.67


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Enc(nn.Module):
    def __init__(self,in_ch,w,d,nl):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        fs=fside(32,d); self.logits=nn.Linear(ch*fs*fs, nl)
    def forward(self,x):
        h=self.net(x).flatten(1); return self.logits(h)

class Dec(nn.Module):
    def __init__(self,out_ch,w,d,nl):
        super().__init__()
        fs=fside(32,d)
        self.fc=nn.Linear(nl, w*fs*fs)
        ups=[]
        for _ in range(d-1): ups+=[nn.ConvTranspose2d(w,w,4,2,1), nn.BatchNorm2d(w), nn.ReLU(True)]
        ups+=[nn.ConvTranspose2d(w,out_ch,4,2,1)]
        self.ups=nn.Sequential(*ups)
        self.w=w; self.fs=fs
    def forward(self,y):
        h=self.fc(y); h=h.view(y.size(0), self.w, self.fs, self.fs)
        return torch.sigmoid(self.ups(h))

class DVAE(nn.Module):
    def __init__(self, cfg: DVAEConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.n_latents)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.n_latents)

    def sample_relaxed_bernoulli(self, logits):
        u = torch.rand_like(logits)
        y = torch.sigmoid((logits + torch.log(u+1e-9) - torch.log(1-u+1e-9)) / self.cfg.tau)
        return y

    def forward(self,x):
        logits=self.enc(x)
        y=self.sample_relaxed_bernoulli(logits)
        xr=self.dec(y)
        return xr, logits, y

    def loss_fn(self,x,xr,logits,y)->Dict[str,torch.Tensor]:
        recon=F.binary_cross_entropy(xr,x,reduction='mean')
        q = torch.sigmoid(logits)
        # KL(q||Bern(0.5)) per-dim
        kl = (q*torch.log((q+1e-9)/0.5) + (1-q)*torch.log(((1-q)+1e-9)/0.5)).mean()
        return {"loss":recon+kl, "recon":recon, "kl":kl}
