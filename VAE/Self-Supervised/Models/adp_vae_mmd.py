from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class MMDVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 128
    depth: int = 2
    lambda_mmd: float = 10.0


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Enc(nn.Module):
    def __init__(self,in_ch,w,d,latent):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        fs=fside(32,d); self.mu=nn.Linear(ch*fs*fs, latent); self.lv=nn.Linear(ch*fs*fs, latent)
    def forward(self,x):
        h=self.net(x); h=h.flatten(1); return self.mu(h), self.lv(h)

class Dec(nn.Module):
    def __init__(self,out_ch,w,d,latent):
        super().__init__()
        fs=fside(32,d); self.fc=nn.Linear(latent, w*fs*fs)
        ups=[]
        for _ in range(d-1): ups+=[nn.ConvTranspose2d(w,w,4,2,1), nn.BatchNorm2d(w), nn.ReLU(True)]
        ups+=[nn.ConvTranspose2d(w,out_ch,4,2,1)]
        self.ups=nn.Sequential(*ups)
    def forward(self,z):
        h=self.fc(z); fs=int((h.numel()//(h.size(0)*self.ups[0].in_channels))**0.5)
        h=h.view(h.size(0), self.ups[0].in_channels, fs, fs)
        return torch.sigmoid(self.ups(h))


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, sigmas=(1, 2, 4, 8, 16)) -> torch.Tensor:
    # x,y: (B,D)
    xx = (x.unsqueeze(1) - x.unsqueeze(0)).pow(2).sum(-1)
    yy = (y.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
    xy = (x.unsqueeze(1) - y.unsqueeze(0)).pow(2).sum(-1)
    mmd = 0.0
    for s in sigmas:
        gamma = 1.0 / (2.0 * (s ** 2))
        k_xx = torch.exp(-gamma * xx)
        k_yy = torch.exp(-gamma * yy)
        k_xy = torch.exp(-gamma * xy)
        mmd += k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return mmd


class MMDVAE(nn.Module):
    def __init__(self, cfg: MMDVAEConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)

    @staticmethod
    def reparameterize(mu, lv):
        std=(0.5*lv).exp(); return mu+torch.randn_like(std)*std

    def forward(self,x):
        mu,lv=self.enc(x); z=self.reparameterize(mu,lv); xr=self.dec(z)
        return xr, mu, lv, z

    def loss_fn(self, x, xr, mu, lv, z) -> Dict[str, torch.Tensor]:
        recon = F.binary_cross_entropy(xr, x, reduction='mean')
        # replace KL with MMD between aggregated q(z) and N(0,I)
        prior = torch.randn_like(z)
        mmd = rbf_mmd(z, prior)
        loss = recon + self.cfg.lambda_mmd * mmd
        # report standard KL for monitoring (not used)
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
        return {"loss": loss, "recon": recon, "mmd": mmd, "kl_monitor": kl}
