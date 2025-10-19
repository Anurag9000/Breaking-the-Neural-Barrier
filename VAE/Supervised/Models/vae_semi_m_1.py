from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# Semi-supervised VAE (M1): VAE + classifier on latent z
# Trains ELBO + supervised CE on labeled data (CIFAR-10 fully labeled here)
# --------------------------------------

@dataclass
class M1Config:
    in_channels: int = 3
    num_classes: int = 10
    latent_dim: int = 64
    width: int = 128
    ce_weight: float = 1.0

class VAE_M1(nn.Module):
    def __init__(self, cfg: M1Config):
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
        self.cls = nn.Linear(cfg.latent_dim, cfg.num_classes)

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = self.reparam(mu, lv)
        x_hat = torch.sigmoid(self.dec(self.fc(z).view(x.size(0), -1, 4, 4)))
        logits = self.cls(mu.detach())  # use mean for stability; detach to avoid interfering with ELBO
        return x_hat, mu, lv, logits

    def loss(self, x, x_hat, mu, lv, y=None):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        kl = -0.5*torch.sum(1 + lv - mu.pow(2) - lv.exp())/x.size(0)
        loss = recon + kl
        ce = torch.tensor(0.0, device=x.device)
        if y is not None:
            logits = self.cls(mu)
            ce = F.cross_entropy(logits, y)
            loss = loss + self.cfg.ce_weight*ce
        return loss, recon.detach(), kl.detach(), ce.detach()
