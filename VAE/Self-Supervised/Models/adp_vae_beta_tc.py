from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BetaTCVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 128
    depth: int = 2
    beta: float = 6.0  # weight on TC term
    mi_weight: float = 1.0
    dimkl_weight: float = 1.0


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class Enc(nn.Module):
    def __init__(self, in_ch,w,d,latent):
        super().__init__()
        ch=w; layers=[nn.Conv2d(in_ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        for _ in range(d-1): layers+=[nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)]
        self.net=nn.Sequential(*layers)
        fs=fside(32,d)
        self.mu=nn.Linear(ch*fs*fs, latent)
        self.lv=nn.Linear(ch*fs*fs, latent)
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


class BetaTCVAE(nn.Module):
    def __init__(self, cfg: BetaTCVAEConfig):
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
        # Reconstruction
        recon=F.binary_cross_entropy(xr, x, reduction='mean')
        # KL decomposition estimation via minibatch estimators (Chen et al., 2018)
        # log q(z|x) is Normal(mu, diag(exp(lv)))
        log_qz_x = (-0.5 * (lv + (z - mu) ** 2 / lv.exp() + torch.log(torch.tensor(2*3.1415926535, device=z.device)))).sum(dim=1)
        # Estimate log q(z) and log(prod_j q(z_j)) using minibatch aggregation
        # compute per-dimension log q(z_j) by averaging over minibatch members' Gaussians
        B, D = z.size()
        # expand to (B,B,D)
        z_exp = z.unsqueeze(1)  # (B,1,D)
        mu_exp = mu.unsqueeze(0)  # (1,B,D)
        lv_exp = lv.unsqueeze(0)  # (1,B,D)
        # log prob under each Gaussian of batch
        log_qz_j = -0.5 * ( (z_exp - mu_exp) ** 2 / lv_exp.exp() + lv_exp + torch.log(torch.tensor(2*3.1415926535, device=z.device)) )  # (B,B,D)
        # log q(z) ~ log mean_k exp(sum_j log q(z_j|x_k))
        log_qz = torch.logsumexp(log_qz_j.sum(dim=2), dim=1) - torch.log(torch.tensor(B, device=z.device, dtype=z.dtype))
        # log prod_j q(z_j) ~ sum_j log mean_k exp(log q(z_j|x_k))
        log_prod_qz = (torch.logsumexp(log_qz_j, dim=1) - torch.log(torch.tensor(B, device=z.device, dtype=z.dtype))).sum(dim=1)
        # log p(z)
        log_pz = (-0.5 * (z**2 + torch.log(torch.tensor(2*3.1415926535, device=z.device)))).sum(dim=1)
        # MI, TC, dim-wise KL
        mi = (log_qz_x - log_qz).mean()
        tc = (log_qz - log_prod_qz).mean()
        dim_kl = (log_prod_qz - log_pz).mean()
        kl_total = mi + tc + dim_kl
        loss = recon + self.cfg.mi_weight * mi + self.cfg.beta * tc + self.cfg.dimkl_weight * dim_kl
        return {"loss": loss, "recon": recon, "kl": kl_total, "mi": mi, "tc": tc, "dimkl": dim_kl}
