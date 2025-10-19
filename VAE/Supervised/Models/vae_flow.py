from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Flow-VAE with Planar flows in q(z|x) (single-model)
# ------------------------------

class Planar(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.u = nn.Parameter(torch.randn(dim)*0.01)
        self.w = nn.Parameter(torch.randn(dim)*0.01)
        self.b = nn.Parameter(torch.zeros(1))

    def forward(self, z):
        # z: (B,D)
        wz = torch.matmul(z, self.w)
        act = torch.tanh(wz + self.b)
        z = z + self.u * act.unsqueeze(1)
        # log-det-J
        psi = (1 - torch.tanh(wz + self.b)**2) * self.w  # (B,D) broadcasting on w
        logdet = torch.log(torch.abs(1 + torch.matmul(psi, self.u)) + 1e-8)
        return z, logdet

@dataclass
class FlowVAEConfig:
    in_channels: int = 3
    latent_dim: int = 32
    width: int = 128
    n_flows: int = 4

class FlowVAE(nn.Module):
    def __init__(self, cfg: FlowVAEConfig):
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
        self.flows = nn.ModuleList([Planar(cfg.latent_dim) for _ in range(cfg.n_flows)])
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
        z0 = self.reparam(mu, lv)
        logdet_sum = 0.0
        z = z0
        for f in self.flows:
            z, ld = f(z)
            logdet_sum = logdet_sum + ld
        h = self.fc(z).view(x.size(0), -1, 4, 4)
        x_hat = torch.sigmoid(self.dec(h))
        return x_hat, mu, lv, z, logdet_sum

    def loss(self, x, x_hat, mu, lv, z, logdet_sum):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        # KL with flows: E_q0[ -log p(zK) + log q0(z0|x) - sum log|det df| ]
        # p(zK)=N(0,I); q0=N(mu, diag(sigma^2))
        log_p = -0.5*(z.pow(2) + torch.log(torch.tensor(2*torch.pi, device=z.device))).sum(dim=1)
        log_q0 = -0.5*(torch.log(2*torch.pi*lv.exp()) + (self.reparam(mu*0+0, lv*0+0) - mu)**2/lv.exp()).sum(dim=1)  # approx constant terms ignored
        kl = (log_q0 - log_p - logdet_sum).mean()
        return recon + kl, recon.detach(), kl.detach()
