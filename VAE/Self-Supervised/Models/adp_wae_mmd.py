from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class WAEMMDConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 128
    depth: int = 2
    lambda_mmd: float = 10.0
    noise_std: float = 0.0  # optional Gaussian noise on z


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Enc(nn.Module):
    def __init__(self,in_ch,w,d,latent):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        fs=fside(32,d); self.fc=nn.Linear(ch*fs*fs, latent)
    def forward(self,x):
        h=self.net(x); h=h.flatten(1); return self.fc(h)

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


def rbf_mmd(x: torch.Tensor, y: torch.Tensor, sigmas=(1,2,4,8,16)) -> torch.Tensor:
    xx=(x.unsqueeze(1)-x.unsqueeze(0)).pow(2).sum(-1)
    yy=(y.unsqueeze(1)-y.unsqueeze(0)).pow(2).sum(-1)
    xy=(x.unsqueeze(1)-y.unsqueeze(0)).pow(2).sum(-1)
    mmd=0.0
    for s in sigmas:
        g=1.0/(2*s*s)
        mmd+=torch.exp(-g*xx).mean()+torch.exp(-g*yy).mean()-2*torch.exp(-g*xy).mean()
    return mmd

class WAEMMD(nn.Module):
    def __init__(self, cfg: WAEMMDConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)

    def forward(self,x):
        z=self.enc(x)
        if self.cfg.noise_std>0:
            z = z + self.cfg.noise_std * torch.randn_like(z)
        xr=self.dec(z)
        return xr, z

    def loss_fn(self, x, xr, z) -> Dict[str, torch.Tensor]:
        recon=F.binary_cross_entropy(xr, x, reduction='mean')
        prior=torch.randn_like(z)
        mmd=rbf_mmd(z, prior)
        loss=recon + self.cfg.lambda_mmd*mmd
        return {"loss":loss, "recon":recon, "mmd":mmd}
