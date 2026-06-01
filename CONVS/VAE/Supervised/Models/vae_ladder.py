from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Ladder VAE (two-level, top-down inference refinement)
# ------------------------------

@dataclass
class LVAEConfig:
    in_channels: int = 3
    z1_dim: int = 32
    z2_dim: int = 32
    width: int = 128

class LVAE(nn.Module):
    def __init__(self, cfg: LVAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        # Bottom-up encoder to produce stats for z1 and z2
        self.enc1 = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_mu2 = nn.Linear(w*4*4*4, cfg.z2_dim)
        self.fc_lv2 = nn.Linear(w*4*4*4, cfg.z2_dim)
        self.fc_mu1_bu = nn.Linear(w*2*8*8, cfg.z1_dim)
        self.fc_lv1_bu = nn.Linear(w*2*8*8, cfg.z1_dim)

        # Top-down prior for z1 conditioned on z2
        self.prior1 = nn.Sequential(
            nn.Linear(cfg.z2_dim, w*2), nn.ReLU(True), nn.Linear(w*2, 2*cfg.z1_dim)
        )
        # Decoder p(x|z1)
        self.dec_fc = nn.Linear(cfg.z1_dim, w*4*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.ConvTranspose2d(w*2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, cfg.in_channels, 4, 2, 1),
        )

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x):
        h1 = self.enc1(x)
        h2 = self.enc2(h1)
        h2f = h2.view(x.size(0), -1)
        mu2, lv2 = self.fc_mu2(h2f), self.fc_lv2(h2f)
        z2 = self.reparam(mu2, lv2)
        # Top-down prior for z1
        m1p, l1p = torch.chunk(self.prior1(z2), 2, dim=1)  # prior mean/logvar for z1
        # Bottom-up proposal for z1
        h1f = h1.view(x.size(0), -1)
        mu1_bu, lv1_bu = self.fc_mu1_bu(h1f), self.fc_lv1_bu(h1f)
        # Refinement: combine prior and bottom-up stats (precision-weighted)
        prec_p = torch.exp(-l1p); prec_bu = torch.exp(-lv1_bu)
        mu1 = (prec_p*m1p + prec_bu*mu1_bu) / (prec_p + prec_bu + 1e-8)
        lv1 = -torch.log(prec_p + prec_bu + 1e-8)
        z1 = self.reparam(mu1, lv1)
        # Decode
        h = self.dec_fc(z1).view(x.size(0), -1, 4, 4)
        x_hat = torch.sigmoid(self.dec(h))
        return x_hat, (mu1, lv1, mu2, lv2)

    def loss(self, x, x_hat, stats):
        mu1, lv1, mu2, lv2 = stats
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        kl2 = -0.5*torch.sum(1 + lv2 - mu2.pow(2) - lv2.exp())/x.size(0)
        kl1 = -0.5*torch.sum(1 + lv1 - mu1.pow(2) - lv1.exp())/x.size(0)
        return recon + kl1 + kl2, recon.detach(), (kl1+kl2).detach()
