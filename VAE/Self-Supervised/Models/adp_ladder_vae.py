from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class LadderVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    z1_dim: int = 32
    z2_dim: int = 16
    width: int = 128
    depth: int = 2


def fside(sz:int,d:int)->int: return max(1, sz//(2**d))

class LadderVAE(nn.Module):
    def __init__(self, cfg: LadderVAEConfig):
        super().__init__()
        self.cfg=cfg
        ch=cfg.width
        # bottom-up encoder producing stats for q(z1|x) and q(z2|z1)
        self.enc1 = nn.Sequential(
            nn.Conv2d(cfg.in_channels,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.Conv2d(ch,ch,3,2,1), nn.BatchNorm2d(ch), nn.ReLU(True)
        )
        fs=fside(32,2)
        self.qz1_mu = nn.Linear(ch*fs*fs, cfg.z1_dim)
        self.qz1_lv = nn.Linear(ch*fs*fs, cfg.z1_dim)
        self.qz2_mu = nn.Linear(cfg.z1_dim, cfg.z2_dim)
        self.qz2_lv = nn.Linear(cfg.z1_dim, cfg.z2_dim)
        # top-down decoder producing p(z1|z2) and p(x|z1)
        self.pz1_mu = nn.Linear(cfg.z2_dim, cfg.z1_dim)
        self.pz1_lv = nn.Linear(cfg.z2_dim, cfg.z1_dim)
        self.dec_fc = nn.Linear(cfg.z1_dim, ch*fs*fs)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(ch,ch,4,2,1), nn.BatchNorm2d(ch), nn.ReLU(True),
            nn.ConvTranspose2d(ch,cfg.in_channels,4,2,1)
        )

    @staticmethod
    def reparameterize(mu, lv):
        std=(0.5*lv).exp(); return mu+torch.randn_like(std)*std

    def forward(self,x):
        h=self.enc1(x); h=h.flatten(1)
        mu1, lv1 = self.qz1_mu(h), self.qz1_lv(h)
        z1 = self.reparameterize(mu1, lv1)
        mu2, lv2 = self.qz2_mu(z1), self.qz2_lv(z1)
        z2 = self.reparameterize(mu2, lv2)
        # top-down prior for z1 given z2
        p_mu1, p_lv1 = self.pz1_mu(z2), self.pz1_lv(z2)
        # decode x from z1
        h = self.dec_fc(z1); fs=fside(32,2); ch=self.cfg.width
        h=h.view(z1.size(0), ch, fs, fs)
        xr=torch.sigmoid(self.dec(h))
        return xr, (mu1, lv1, mu2, lv2, p_mu1, p_lv1)

    def loss_fn(self, x, xr, stats) -> Dict[str, torch.Tensor]:
        mu1, lv1, mu2, lv2, p_mu1, p_lv1 = stats
        recon=F.binary_cross_entropy(xr, x, reduction='mean')
        # KL(z2 || N(0,I)) + KL(z1 || p(z1|z2))
        kl_z2 = -0.5 * torch.sum(1 + lv2 - mu2.pow(2) - lv2.exp(), dim=1).mean()
        # KL for two diagonal Gaussians
        var1 = lv1.exp(); p_var1 = p_lv1.exp()
        kl_z1 = 0.5 * ( ( (var1 / p_var1) + (mu1 - p_mu1).pow(2) / p_var1 - 1 + (p_lv1 - lv1) ).sum(dim=1) ).mean()
        loss = recon + kl_z1 + kl_z2
        return {"loss":loss, "recon":recon, "kl_z1":kl_z1, "kl_z2":kl_z2}
