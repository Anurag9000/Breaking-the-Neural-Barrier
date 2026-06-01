from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# β-TCVAE (Total Correlation penalty via minibatch-weighted estimators)
# Reference-style implementation (no discriminator)
# ------------------------------

@dataclass
class BTCVAEConfig:
    in_channels: int = 3
    latent_dim: int = 32
    width: int = 128
    beta: float = 6.0  # weight on Total Correlation term
    gamma: float = 1.0 # weight on dim-wise KL (optional)

class BTCVAE(nn.Module):
    def __init__(self, cfg: BTCVAEConfig):
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

    # Minibatch-weighted estimators of log q(z) and log q(z_j)
    # See: Chen et al. 2018 (β-TCVAE) appendix
    def _log_density_gaussian(self, z, mu, logvar):
        # z: (B,D), mu/logvar: (B,D)
        D = z.size(1)
        log2pi = torch.log(torch.tensor(2*torch.pi, device=z.device))
        return -0.5*(D*log2pi + ((z-mu)**2)/logvar.exp() + logvar.sum(dim=1, keepdim=False))

    def _matrix_log_q_z(self, z, mu, logvar):
        # pairwise log prob of z under each param row
        # returns (B,B)
        B, D = z.size()
        z_expand = z.view(B, 1, D)
        mu = mu.view(1, B, D)
        logvar = logvar.view(1, B, D)
        log2pi = torch.log(torch.tensor(2*torch.pi, device=z.device))
        # (B,B,D)
        term = -0.5*(log2pi + logvar + (z_expand - mu)**2 / logvar.exp())
        return term.sum(dim=2)

    def btcvae_loss(self, x, x_hat, mu, logvar, z):
        B, D = z.size()
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/B
        # Compute TC and dim-wise KL with minibatch weights
        log_q_zx = -0.5*(torch.log(2*torch.pi*logvar.exp()) + (z-mu)**2/logvar.exp()).sum(dim=1)  # (B,)
        mat_log_qz = self._matrix_log_q_z(z, mu, logvar)  # (B,B)
        # log q(z) via log-sum-exp over batch
        log_q_z = torch.logsumexp(mat_log_qz, dim=1) - torch.log(torch.tensor(B, device=z.device, dtype=z.dtype))
        # dim-wise log q(z_j)
        mat_log_qzj = mat_log_qz.unsqueeze(2)  # (B,B,1)
        # approximate per-dim by reusing same params (diagonal Gaussian assumption) → sum over dims later
        # For TC we need: KL( q(z) || Π_j q(z_j) ) = E[log q(z) - Σ_j log q(z_j)]
        # Approximate Σ_j log q(z_j) by sum over dimensions using moments from minibatch
        # Here we fallback to diag-Gaussian analytic per-dim q(z_j) ~ N(μ_j, σ_j^2) averaged over batch
        mu_bar = mu.mean(dim=0)
        var_bar = logvar.exp().mean(dim=0)
        log_q_zj = (-0.5*(torch.log(2*torch.pi*var_bar) + (z - mu_bar)**2/var_bar)).sum(dim=1)
        tc = (log_q_z - log_q_zj).mean()
        # Dim-wise KL to N(0,1)
        kl_dim = 0.5*(mu.pow(2) + logvar.exp() - logvar - 1.0).mean(dim=0).sum()
        loss = recon + self.cfg.beta*tc + self.cfg.gamma*kl_dim
        return loss, recon.detach(), tc.detach(), kl_dim.detach()
