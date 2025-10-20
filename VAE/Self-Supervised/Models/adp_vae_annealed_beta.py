from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class AnnealedBetaConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 128
    depth: int = 2
    beta_start: float = 0.0
    beta_end: float = 4.0
    warmup_epochs: int = 10


def fside(sz:int, d:int)->int: return max(1, sz//(2**d))


class Enc(nn.Module):
    def __init__(self, in_ch, width, depth, latent):
        super().__init__()
        ch=width; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(depth-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        fs=fside(32,depth)
        self.mu=nn.Linear(ch*fs*fs, latent)
        self.lv=nn.Linear(ch*fs*fs, latent)
    def forward(self,x):
        h=self.net(x); h=h.flatten(1); return self.mu(h), self.lv(h)

class Dec(nn.Module):
    def __init__(self, out_ch, width, depth, latent):
        super().__init__()
        fs=fside(32,depth)
        self.fc=nn.Linear(latent, width*fs*fs)
        ups=[]
        for _ in range(depth-1): ups+=[nn.ConvTranspose2d(width,width,4,2,1), nn.BatchNorm2d(width), nn.ReLU(True)]
        ups+=[nn.ConvTranspose2d(width,out_ch,4,2,1)]
        self.ups=nn.Sequential(*ups)
    def forward(self,z):
        h=self.fc(z); fs=int((h.numel()//(h.size(0)*self.ups[0].in_channels))**0.5)
        h=h.view(h.size(0), self.ups[0].in_channels, fs, fs)
        return torch.sigmoid(self.ups(h))


class AnnealedBetaVAE(nn.Module):
    def __init__(self, cfg: AnnealedBetaConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self._beta=cfg.beta_start

    def encode(self,x): return self.enc(x)
    @staticmethod
    def reparameterize(mu,lv):
        std=(0.5*lv).exp(); return mu+torch.randn_like(std)*std
    def decode(self,z): return self.dec(z)
    def forward(self,x):
        mu,lv=self.encode(x); z=self.reparameterize(mu,lv); xr=self.decode(z); return xr,mu,lv

    def set_beta_for_epoch(self, epoch:int):
        if epoch>=self.cfg.warmup_epochs:
            self._beta=self.cfg.beta_end
        else:
            t=epoch/max(1,self.cfg.warmup_epochs)
            self._beta=self.cfg.beta_start + t*(self.cfg.beta_end-self.cfg.beta_start)

    def loss_fn(self, x, xr, mu, lv) -> Dict[str, torch.Tensor]:
        recon=F.binary_cross_entropy(xr, x, reduction='mean')
        kl= -0.5*torch.sum(1+lv - mu.pow(2)-lv.exp(), dim=1).mean()
        loss=recon + self._beta*kl
        return {"loss":loss, "recon":recon, "kl":kl, "beta": torch.tensor(self._beta, device=x.device)}
