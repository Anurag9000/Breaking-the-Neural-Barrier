from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class PlanarFlowVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 128
    depth: int = 2
    n_flows: int = 4


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

class PlanarFlow(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim))
        self.w = nn.Parameter(torch.randn(dim))
        self.b = nn.Parameter(torch.zeros(1))
    def forward(self, z):
        # z: (B,D)
        lin = z @ self.w + self.b  # (B,)
        h = torch.tanh(lin)
        z_ = z + self.u * h.unsqueeze(-1)
        # log|det J| = log|1 + u^T psi|, psi = (1 - tanh^2(lin)) * w
        psi = (1 - h.pow(2)).unsqueeze(-1) * self.w
        log_det = torch.log(torch.abs(1 + (psi * self.u).sum(dim=-1)) + 1e-8)
        return z_, log_det

class PlanarFlowVAE(nn.Module):
    def __init__(self, cfg: PlanarFlowVAEConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.flows = nn.ModuleList([PlanarFlow(cfg.latent_dim) for _ in range(cfg.n_flows)])

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
        # KL with flow: KL = E[log q0(z0|x) - log p(zK) - sum log|det|]
        z0_mu, z0_lv = mu, lv
        kl0 = -0.5*torch.sum(1+z0_lv - z0_mu.pow(2) - z0_lv.exp(), dim=1)
        kl = (kl0 - sum_ld).mean()
        return {"loss": recon+kl, "recon":recon, "kl":kl}
