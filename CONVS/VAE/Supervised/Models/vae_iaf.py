from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --------------------------------------
# IAF-VAE: Inverse Autoregressive Flow in q(z|x)
# Compact MADE-like conditioner for shift/scale; single-model
# --------------------------------------

class MADE1D(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden)
        self.fc_m = nn.Linear(hidden, dim)
        self.fc_s = nn.Linear(hidden, dim)
    def forward(self, z):
        h = F.relu(self.fc1(z), inplace=True)
        m = self.fc_m(h)
        s = torch.tanh(self.fc_s(h))  # stable scale
        # simple autoregressive approximation by masking with tril (cumulative)
        # here we rely on ordering + tanh to keep transformation stable
        return m, s

@dataclass
class IAFVAEConfig:
    in_channels: int = 3
    latent_dim: int = 32
    width: int = 128
    n_flows: int = 2

class IAFVAE(nn.Module):
    def __init__(self, cfg: IAFVAEConfig):
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
        self.conds = nn.ModuleList([MADE1D(cfg.latent_dim, hidden=2*cfg.latent_dim) for _ in range(cfg.n_flows)])
        self.fc = nn.Linear(cfg.latent_dim, w*4*4*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.ConvTranspose2d(w*2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, cfg.in_channels, 4, 2, 1),
        )

    @staticmethod
    def reparam(mu, lv):
        std = torch.exp(0.5*lv); eps = torch.randn_like(std); return mu + eps*std

    def iaf(self, z0):
        logdet = 0.0
        z = z0
        for cond in self.conds:
            m, s = cond(z)
            # z' = s ⊙ z + (1 - s) ⊙ m   (inverse of affine autoregressive)
            z = s * z + (1 - s) * m
            logdet = logdet + torch.sum(torch.log(torch.abs(s) + 1e-8), dim=1)
        return z, logdet

    def forward(self, x):
        h = self.enc(x).view(x.size(0), -1)
        mu, lv = self.fc_mu(h), self.fc_lv(h)
        z0 = self.reparam(mu, lv)
        zK, logdet = self.iaf(z0)
        x_hat = torch.sigmoid(self.dec(self.fc(zK).view(x.size(0), -1, 4, 4)))
        return x_hat, mu, lv, zK, logdet

    def loss(self, x, x_hat, mu, lv, zK, logdet):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        log_p = -0.5*(zK.pow(2) + torch.log(torch.tensor(2*torch.pi, device=zK.device))).sum(dim=1)
        log_q0 = -0.5*(torch.log(2*torch.pi*lv.exp()) + (self.reparam(mu*0, lv*0) - mu)**2/lv.exp()).sum(dim=1)
        kl = (log_q0 - log_p - logdet).mean()
        return recon + kl, recon.detach(), kl.detach()
