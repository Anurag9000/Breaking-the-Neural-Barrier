from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# Spatial Broadcast Decoder VAE (SBD-VAE)
# Decoder receives z broadcast across spatial grid + (x,y) coord channels
# --------------------------------------

@dataclass
class SBDConfig:
    in_channels: int = 3
    latent_dim: int = 16
    width: int = 64

class SBDVAE(nn.Module):
    def __init__(self, cfg: SBDConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        # Encoder: simple downsampling convs to a latent
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(w, w*2, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 4, 2, 1), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc_lv = nn.Linear(w*4*4*4, cfg.latent_dim)
        # Decoder: convs over broadcasted z + coords => image
        self.dec = nn.Sequential(
            nn.Conv2d(cfg.latent_dim + 2, w, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(w, w, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(w, w, 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(w, cfg.in_channels, 1)
        )

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def spatial_broadcast(self, z, size=32):
        B, D = z.shape
        z = z.view(B, D, 1, 1).expand(B, D, size, size)
        # coord channels in [-1,1]
        xs = torch.linspace(-1, 1, size, device=z.device).view(1,1,1,size).expand(B,1,size,size)
        ys = torch.linspace(-1, 1, size, device=z.device).view(1,1,size,1).expand(B,1,size,size)
        return torch.cat([z, xs, ys], dim=1)

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = self.reparam(mu, lv)
        feat = self.spatial_broadcast(z, size=32)
        x_hat = torch.sigmoid(self.dec(feat))
        return x_hat, mu, lv

    def loss(self, x, x_hat, mu, lv):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        kl = -0.5*torch.sum(1 + lv - mu.pow(2) - lv.exp())/x.size(0)
        return recon + kl, recon.detach(), kl.detach()
