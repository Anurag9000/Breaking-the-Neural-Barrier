from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class IAFVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 128
    depth: int = 2
    n_flows: int = 2
    hidden: int = 64


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Enc(nn.Module):
    def __init__(self,in_ch,w,d,latent):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        fs=fside(32,d); self.mu=nn.Linear(ch*fs*fs, latent); self.lv=nn.Linear(ch*fs*fs, latent)
    def forward(self,x):
        h=self.net(x).flatten(1); return self.mu(h), self.lv(h)

class Dec(nn.Module):
    def __init__(self,out_ch,w,d,latent):
        super().__init__()
        fs=fside(32,d)
        self.fc=nn.Linear(latent, w*fs*fs)
        ups=[]
        for _ in range(d-1): ups+=[nn.ConvTranspose2d(w,w,4,2,1), nn.BatchNorm2d(w), nn.ReLU(True)]
        ups+=[nn.ConvTranspose2d(w,out_ch,4,2,1)]
        self.ups=nn.Sequential(*ups)
        self.w=w; self.fs=fs
    def forward(self,z):
        h=self.fc(z); h=h.view(z.size(0), self.w, self.fs, self.fs)
        return torch.sigmoid(self.ups(h))

class MADE(nn.Module):
    # simple fully-connected MADE for autoregressive parameters
    def __init__(self, dim, hidden):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc_m = nn.Linear(hidden, dim)
        self.fc_s = nn.Linear(hidden, dim)
    def forward(self, z):
        h = F.relu(self.fc1(z))
        m = self.fc_m(h)
        s = torch.tanh(self.fc_s(h))  # scale in (-1,1)
        return m, s

class IAF(nn.Module):
    def __init__(self, dim, hidden):
        super().__init__()
        self.net = MADE(dim, hidden)
    def forward(self, z):
        m, s = self.net(z)
        z_ = s * z + (1 - s) * m
        # log det = sum log|s|
        log_det = torch.sum(torch.log(torch.abs(s) + 1e-8), dim=1)
        return z_, log_det

class IAFVAE(nn.Module):
    def __init__(self, cfg: IAFVAEConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.flows = nn.ModuleList([IAF(cfg.latent_dim, cfg.hidden) for _ in range(cfg.n_flows)])

    @staticmethod
    def reparameterize(mu, lv):
        std=(0.5*lv).exp(); return mu + torch.randn_like(std)*std

    def forward(self,x):
        mu,lv=self.enc(x); z=self.reparameterize(mu,lv); sum_log_det=0.0
        for f in self.flows:
            z, ld = f(z); sum_log_det = sum_log_det + ld
        xr=self.dec(z)
        return xr, mu, lv, sum_log_det

    def loss_fn(self,x,xr,mu,lv,sum_ld) -> Dict[str,torch.Tensor]:
        recon=F.binary_cross_entropy(xr,x,reduction='mean')
        kl0 = -0.5*torch.sum(1+lv - mu.pow(2) - lv.exp(), dim=1)
        kl = (kl0 - sum_ld).mean()
        return {"loss": recon+kl, "recon":recon, "kl":kl}
