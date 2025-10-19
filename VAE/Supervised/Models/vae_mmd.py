from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# MMD-VAE / InfoVAE (single-model, MMD penalty instead of KL)
# ------------------------------

def rbf_kernel(x, y, sigma=1.0):
    x = x.unsqueeze(1)  # (B,1,D)
    y = y.unsqueeze(0)  # (1,B,D)
    diff = x - y
    dist_sq = (diff*diff).sum(-1)
    k = torch.exp(-dist_sq / (2*sigma*sigma))
    return k

@torch.no_grad()
def mmd_loss(z, prior_samples, sigma=1.0):
    k_xx = rbf_kernel(z, z, sigma)
    k_yy = rbf_kernel(prior_samples, prior_samples, sigma)
    k_xy = rbf_kernel(z, prior_samples, sigma)
    m = z.size(0)
    # Unbiased MMD^2
    mmd2 = (k_xx.sum() - k_xx.diag().sum())/(m*(m-1)) \
         + (k_yy.sum() - k_yy.diag().sum())/(m*(m-1)) \
         - 2*k_xy.mean()
    return mmd2

@dataclass
class MMDVAEConfig:
    in_channels: int = 3
    latent_dim: int = 64
    width: int = 128
    sigma: float = 2.0
    recon_weight: float = 1.0
    mmd_weight: float = 10.0

class MMDVAE(nn.Module):
    def __init__(self, cfg: MMDVAEConfig):
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
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu, logvar = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparam(mu, logvar)
        h = self.fc(z).view(x.size(0), -1, 4, 4)
        x_hat = torch.sigmoid(self.dec(h))
        return x_hat, z

    def loss(self, x, x_hat, z):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        prior_samp = torch.randn_like(z)
        mmd = mmd_loss(z, prior_samp, sigma=self.cfg.sigma)
        loss = self.cfg.recon_weight*recon + self.cfg.mmd_weight*mmd
        return loss, recon.detach(), mmd.detach()
