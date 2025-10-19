import math
from dataclasses import dataclass
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Vanilla VAE (ELBO)
# ------------------------------

@dataclass
class VAEConfig:
    in_channels: int = 3
    latent_dim: int = 64
    width: int = 128  # base conv channels
    img_size: int = 32

class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, width: int, latent_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, 2, 1), nn.BatchNorm2d(width), nn.ReLU(True),
            nn.Conv2d(width, width*2, 3, 2, 1), nn.BatchNorm2d(width*2), nn.ReLU(True),
            nn.Conv2d(width*2, width*4, 3, 2, 1), nn.BatchNorm2d(width*4), nn.ReLU(True),
        )
        self.out_ch = width*4
        self.fc_mu = nn.Linear(self.out_ch*4*4, latent_dim)
        self.fc_logvar = nn.Linear(self.out_ch*4*4, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        h = h.view(h.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class ConvDecoder(nn.Module):
    def __init__(self, out_ch: int, width: int, latent_dim: int):
        super().__init__()
        self.fc = nn.Linear(latent_dim, width*4*4*4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(width*4, width*2, 4, 2, 1), nn.BatchNorm2d(width*2), nn.ReLU(True),
            nn.ConvTranspose2d(width*2, width, 4, 2, 1), nn.BatchNorm2d(width), nn.ReLU(True),
            nn.ConvTranspose2d(width, out_ch, 4, 2, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        x_recon = self.net(h)
        return torch.sigmoid(x_recon)

class VAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = ConvEncoder(cfg.in_channels, cfg.width, cfg.latent_dim)
        self.dec = ConvDecoder(cfg.in_channels, cfg.width, cfg.latent_dim)

    @staticmethod
    def reparam(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x: torch.Tensor):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar

    def elbo_loss(self, x: torch.Tensor, x_hat: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        # Reconstruction: BCE (pixelwise)
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum') / x.size(0)
        # KL(q||p) for N(mu, sigma^2) vs N(0,1)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.size(0)
        return recon + kl, recon.detach(), kl.detach()
