from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# VAMPPrior VAE (learnable pseudo-inputs as mixture prior)
# ------------------------------

@dataclass
class VampConfig:
    in_channels: int = 3
    latent_dim: int = 64
    width: int = 128
    K: int = 500  # number of pseudo-inputs

class Encoder(nn.Module):
    def __init__(self, in_ch, w, latent_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, w, 3, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.Conv2d(w, w*2, 3, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.Conv2d(w*2, w*4, 3, 2, 1), nn.BatchNorm2d(w*4), nn.ReLU(True),
        )
        self.fc_mu = nn.Linear(w*4*4*4, latent_dim)
        self.fc_logvar = nn.Linear(w*4*4*4, latent_dim)

    def forward(self, x):
        h = self.net(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

class Decoder(nn.Module):
    def __init__(self, out_ch, w, latent_dim):
        super().__init__()
        self.fc = nn.Linear(latent_dim, w*4*4*4)
        self.net = nn.Sequential(
            nn.ConvTranspose2d(w*4, w*2, 4, 2, 1), nn.BatchNorm2d(w*2), nn.ReLU(True),
            nn.ConvTranspose2d(w*2, w, 4, 2, 1), nn.BatchNorm2d(w), nn.ReLU(True),
            nn.ConvTranspose2d(w, out_ch, 4, 2, 1),
        )

    def forward(self, z):
        h = self.fc(z).view(z.size(0), -1, 4, 4)
        return torch.sigmoid(self.net(h))

class VampPriorVAE(nn.Module):
    def __init__(self, cfg: VampConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(cfg.in_channels, cfg.width, cfg.latent_dim)
        self.dec = Decoder(cfg.in_channels, cfg.width, cfg.latent_dim)
        # Pseudo-inputs as learnable parameters in pixel space [0,1]
        self.pseudo = nn.Parameter(torch.rand(cfg.K, cfg.in_channels, 32, 32))

    @staticmethod
    def reparam(mu, logvar):
        std = torch.exp(0.5*logvar); eps = torch.randn_like(std); return mu + eps*std

    def prior_params(self):
        with torch.no_grad():
            xk = self.pseudo.clamp(0,1)
        mu_k, logvar_k = self.enc(xk)
        return mu_k, logvar_k  # (K,D)

    def forward(self, x):
        mu, logvar = self.enc(x)
        z = self.reparam(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar, z

    def loss(self, x, x_hat, mu, logvar, z):
        recon = F.binary_cross_entropy(x_hat, x, reduction='sum')/x.size(0)
        # KL to VAMP prior (mixture of encodings of pseudo-inputs)
        mu_k, logvar_k = self.prior_params()  # (K,D)
        # log p(z) = log(1/K sum_k N(z|mu_k, sigma_k))
        z_exp = z.unsqueeze(1)  # (B,1,D)
        mu_k = mu_k.unsqueeze(0)  # (1,K,D)
        logvar_k = logvar_k.unsqueeze(0)  # (1,K,D)
        log_prob_k = -0.5*(torch.log(2*torch.pi*logvar_k.exp()) + (z_exp-mu_k)**2/logvar_k.exp()).sum(dim=2)  # (B,K)
        log_pz = torch.logsumexp(log_prob_k - math.log(self.cfg.K), dim=1)  # (B,)
        # log q(z|x): diagonal Gaussian
        log_qzx = -0.5*(torch.log(2*torch.pi*logvar.exp()) + (z-mu)**2/logvar.exp()).sum(dim=1)
        kl = (log_qzx - log_pz).mean()
        return recon + kl, recon.detach(), kl.detach()
