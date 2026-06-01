from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# WAE-MMD (single-model) with deterministic encoder
# Loss = recon(x, dec(enc(x))) + lambda * MMD(z, prior)
# ------------------------------

@dataclass
class WAEConfig:
    in_channels: int = 3
    latent_dim: int = 64
    width: int = 128
    mmd_weight: float = 10.0
    sigma: float = 2.0


def rbf_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)
    y = y.unsqueeze(0)
    diff = x - y
    dist_sq = (diff*diff).sum(-1)
    return torch.exp(-dist_sq/(2*sigma*sigma))

@torch.no_grad()
def mmd(z, prior, sigma):
    k_xx = rbf_kernel(z, z, sigma)
    k_yy = rbf_kernel(prior, prior, sigma)
    k_xy = rbf_kernel(z, prior, sigma)
    m = z.size(0)
    return (k_xx.sum()-k_xx.diag().sum())/(m*(m-1)) + (k_yy.sum()-k_yy.diag().sum())/(m*(m-1)) - 2*k_xy.mean()

class WAE_MMD(nn.Module):
    def __init__(self, cfg: WAEConfig):
        super().__init__()
        self.cfg = cfg
        w = cfg.width
        self.enc = nn.Sequential(
            nn.Conv2d(cfg.in_channels, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_z = nn.Linear(w*4*4*4, cfg.latent_dim)
        self.fc = nn.Linear(cfg.latent_dim, w*4*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.ConvTranspose2d(w*2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, cfg.in_channels, 4, 2, 1),
        )

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        z = self.fc_z(h)
        x_hat = torch.sigmoid(self.dec(self.fc(z).view(x.size(0), -1, 4, 4)))
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        prior = torch.randn_like(z)
        penalty = mmd(z, prior, self.cfg.sigma)
        return recon + self.cfg.mmd_weight*penalty, recon.detach(), penalty.detach()
