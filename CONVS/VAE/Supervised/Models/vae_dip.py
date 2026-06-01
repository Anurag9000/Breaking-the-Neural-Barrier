from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# DIP-VAE (I/II) — Regularize covariance of inferred latents
# We implement DIP-VAE-I style: encourage Cov_z to match I via penalties on moments
# ------------------------------

@dataclass
class DIPVAEConfig:
    in_channels: int = 3
    latent_dim: int = 32
    width: int = 128
    lambda_diag: float = 10.0
    lambda_offdiag: float = 5.0

class DIPVAE(nn.Module):
    def __init__(self, cfg: DIPVAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc_logvar = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc = nn.Linear(cfg.latent_dim, w*4*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.ConvTranspose2d(w*2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, cfg.in_channels, 4, 2, 1),
        )

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5*logvar); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparam(mu, logvar)
        h = self.fc(z).view(x.size(0), -1, 4, 4)
        x_hat = torch.sigmoid(self.dec(h))
        return x_hat, mu, logvar, z

    def dip_regularizer(self, mu):
        # DIP-VAE-I uses the covariance of the inferred means across the batch
        B, D = mu.size()
        mu_centered = mu - mu.mean(dim=0, keepdim=True)
        cov = (mu_centered.t() @ mu_centered) / (B - 1 + 1e-8)  # (D,D)
        # match to identity: penalize off-diagonal and (diag - 1)^2
        offdiag = cov - torch.diag(torch.diag(cov))
        reg = self.cfg.lambda_offdiag * (offdiag**2).sum() + \
              self.cfg.lambda_diag * ((torch.diag(cov) - 1.0)**2).sum()
        return reg / B

    def loss(self, x, x_hat, mu, logvar):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())/x.size(0)
        dip = self.dip_regularizer(mu)
        return recon + kl + dip, recon.detach(), kl.detach(), dip.detach()
