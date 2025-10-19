from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# Delta-VAE (δ-VAE): encourage non-zero KL by targeting a minimum KL per-dim
# Implemented via free-bits style: clamp per-dim KL below delta.
# --------------------------------------

@dataclass
class DeltaVAEConfig:
    in_channels: int = 3
    latent_dim: int = 32
    width: int = 128
    delta: float = 0.5  # minimum KL per-dimension (nats)

class DeltaVAE(nn.Module):
    def __init__(self, cfg: DeltaVAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc_lv = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc = nn.Linear(cfg.latent_dim, w*4*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.ConvTranspose2d(w*2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, cfg.in_channels, 4, 2, 1),
        )

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = self.reparam(mu, lv)
        x_hat = torch.sigmoid(self.dec(self.fc(z).view(x.size(0), -1, 4, 4)))
        return x_hat, mu, lv

    def loss(self, x, x_hat, mu, lv):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        # per-dim KL
        kl_dim = 0.5*(mu.pow(2) + lv.exp() - lv - 1.0)  # (B,D)
        free_bits = torch.clamp(kl_dim.mean(0), min=self.cfg.delta)  # (D,)
        kl = free_bits.sum()
        return recon + kl, recon.detach(), kl.detach()
