from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# Conditional VAE (CVAE): condition on class y via learned embedding
# --------------------------------------

@dataclass
class CVAEConfig:
    in_channels: int = 3
    num_classes: int = 10
    latent_dim: int = 64
    width: int = 128
    emb_dim: int = 32

class CVAE(nn.Module):
    def __init__(self, cfg: CVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.emb = nn.Embedding(cfg.num_classes, cfg.emb_dim)
        w = cfg.width
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels + 1, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc_lv = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc = nn.Linear(cfg.latent_dim + cfg.emb_dim, w*4*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.ConvTranspose2d(w*2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, cfg.in_channels, 4, 2, 1),
        )

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x, y):
        B, _, H, W = x.shape
        ey = self.emb(y)
        # tile class channel (one channel with class id normalized)
        c = (y.float()/ (self.cfg.num_classes-1)).view(B,1,1,1).expand(B,1,H,W)
        h = self.enc(torch.cat([x, c], dim=1)).view(B, -1)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = self.reparam(mu, lv)
        h = self.fc(torch.cat([z, ey], dim=1)).view(B, -1, 4, 4)
        x_hat = torch.sigmoid(self.dec(h))
        return x_hat, mu, lv

    def loss(self, x, x_hat, mu, lv):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        kl = -0.5*torch.sum(1 + lv - mu.pow(2) - lv.exp())/x.size(0)
        return recon + kl, recon.detach(), kl.detach()
