from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# Mixture-of-Gaussians Prior VAE (learnable mixture weights/means/vars)
# --------------------------------------

@dataclass
class MoGPriorConfig:
    in_channels: int = 3
    latent_dim: int = 32
    width: int = 128
    K: int = 10

class MoGPriorVAE(nn.Module):
    def __init__(self, cfg: MoGPriorConfig):
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
        # learnable mixture prior params
        self.pi_logits = nn.Parameter(torch.zeros(cfg.K))
        self.mu_k = nn.Parameter(torch.randn(cfg.K, cfg.latent_dim)*0.1)
        self.logvar_k = nn.Parameter(torch.zeros(cfg.K, cfg.latent_dim))

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z = self.reparam(mu, lv)
        x_hat = torch.sigmoid(self.dec(self.fc(z).view(x.size(0), -1, 4, 4)))
        return x_hat, mu, lv, z

    def log_mix_prob(self, z):
        # log p(z) under mixture
        pi = torch.softmax(self.pi_logits, dim=0)  # (K,)
        z = z.unsqueeze(1)  # (B,1,D)
        mu = self.mu_k.unsqueeze(0)
        lv = self.logvar_k.unsqueeze(0)
        log_comp = -0.5*(torch.log(2*torch.pi*lv.exp()) + (z-mu)**2/lv.exp()).sum(-1)  # (B,K)
        return torch.logsumexp(torch.log(pi.unsqueeze(0)) + log_comp, dim=1)  # (B,)

    def loss(self, x, x_hat, mu, lv, z):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        log_q = -0.5*(torch.log(2*torch.pi*lv.exp()) + (z-mu)**2/lv.exp()).sum(-1).mean()
        log_p = self.log_mix_prob(z).mean()
        kl = (log_q - log_p)
        return recon + kl, recon.detach(), kl.detach()
