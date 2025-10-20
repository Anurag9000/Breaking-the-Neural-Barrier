from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class DIPVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 128
    depth: int = 2
    lambda_diag: float = 10.0
    lambda_offdiag: float = 5.0
    target_var: float = 1.0  # encourage unit variance


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Enc(nn.Module):
    def __init__(self, in_ch,w,d,latent):
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


class DIPVAE(nn.Module):
    def __init__(self, cfg: DIPVAEConfig):
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

    def _covariance_penalty(self, mu_batch: torch.Tensor) -> torch.Tensor:
        # Centered covariance of encoder means over batch
        B, D = mu_batch.size()
        mu_centered = mu_batch - mu_batch.mean(dim=0, keepdim=True)
        cov = (mu_centered.t() @ mu_centered) / (B - 1 + 1e-6)  # (D,D)
        diag = torch.diagonal(cov)
        offdiag = cov - torch.diag(diag)
        diag_pen = (diag - self.cfg.target_var).pow(2).sum()
        off_pen = (offdiag.pow(2)).sum()
        return self.cfg.lambda_diag * diag_pen + self.cfg.lambda_offdiag * off_pen

    def loss_fn(self, x, xr, mu, lv, z) -> Dict[str, torch.Tensor]:
        recon = F.binary_cross_entropy(xr, x, reduction='mean')
        kl = -0.5 * torch.sum(1 + lv - mu.pow(2) - lv.exp(), dim=1).mean()
        dip = self._covariance_penalty(mu)
        loss = recon + kl + dip
        return {"loss": loss, "recon": recon, "kl": kl, "dip": dip}
