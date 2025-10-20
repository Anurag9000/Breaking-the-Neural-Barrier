from dataclasses import dataclass
from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VMFVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 16  # vMF dimension
    width: int = 128
    depth: int = 2
    kappa_min: float = 1.0


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Enc(nn.Module):
    def __init__(self,in_ch,w,d,latent):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        fs=fside(32,d); self.mu = nn.Linear(ch*fs*fs, latent)   # raw mean direction
        self.kappa = nn.Linear(ch*fs*fs, 1)                     # concentration >0
    def forward(self,x):
        h=self.net(x); h=h.flatten(1)
        mu = F.normalize(self.mu(h), dim=-1)
        kappa = F.softplus(self.kappa(h)) + 1e-3
        return mu, kappa.squeeze(-1)

class Dec(nn.Module):
    def __init__(self,out_ch,w,d,latent):
        super().__init__()
        fs=fside(32,d); self.fc=nn.Linear(latent, w*fs*fs)
        ups=[]
        for _ in range(d-1): ups+=[nn.ConvTranspose2d(w,w,4,2,1), nn.BatchNorm2d(w), nn.ReLU(True)]
        ups+=[nn.ConvTranspose2d(w,out_ch,4,2,1)]
        self.ups=nn.Sequential(*ups)
        self.w=w; self.fs=fs
    def forward(self,z):
        h=self.fc(z); h=h.view(z.size(0), self.w, self.fs, self.fs)
        return torch.sigmoid(self.ups(h))


def sample_vmf(mu: torch.Tensor, kappa: torch.Tensor) -> torch.Tensor:
    # Rejection sampling approximation for vMF on S^{d-1}
    # For practicality, use simple Gaussian noise + normalization when kappa small
    B, D = mu.size()
    eps = torch.randn_like(mu)
    z = mu + eps / (kappa.unsqueeze(-1) + 1.0)
    return F.normalize(z, dim=-1)


def vmf_kl_to_uniform(kappa: torch.Tensor, d: int) -> torch.Tensor:
    # Approximate KL between vMF(mu,kappa) and uniform on sphere; use bound ~ -(H_vMF - H_uniform)
    # Use entropy approximation H_vMF ~ something monotone decreasing with kappa; here simple proxy
    # This term encourages small kappa (spread), akin to spherical prior.
    return (kappa.clamp_min(0.0) / (d)).mean()

class VMFVAE(nn.Module):
    def __init__(self, cfg: VMFVAEConfig):
        super().__init__()
        self.cfg=cfg
        self.enc=Enc(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)
        self.dec=Dec(cfg.in_channels,cfg.width,cfg.depth,cfg.latent_dim)

    def forward(self,x):
        mu, kappa = self.enc(x)
        kappa = kappa.clamp_min(self.cfg.kappa_min)
        z = sample_vmf(mu, kappa)
        xr = self.dec(z)
        return xr, mu, kappa, z

    def loss_fn(self, x, xr, mu, kappa, z) -> Dict[str, torch.Tensor]:
        recon = F.binary_cross_entropy(xr, x, reduction='mean')
        kl = vmf_kl_to_uniform(kappa, self.cfg.latent_dim)
        loss = recon + kl
        return {"loss":loss, "recon":recon, "kl":kl}
