import torch
import torch.nn as nn
import torch.nn.functional as F

# Discretized Logistic Likelihood VAE (for 0-1 scaled images, 8-bit bins)
# log p(x|z) = log( Sigmoid((x+Δ/2 - μ)/s) - Sigmoid((x-Δ/2 - μ)/s) )
# We work with Δ=1/256 and clamp the CDF difference for stability.

class Encoder(nn.Module):
    def __init__(self, in_ch=1, z_dim=32):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(32, 64, 3, 2, 1), nn.ReLU(True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(True),
        )
        self.mu = nn.Linear(128*4*4, z_dim)
        self.lv = nn.Linear(128*4*4, z_dim)
    def forward(self, x):
        h = self.conv(x).view(x.size(0), -1)
        return self.mu(h), self.lv(h)

class Decoder(nn.Module):
    def __init__(self, out_ch=1, z_dim=32):
        super().__init__()
        self.fc = nn.Linear(z_dim, 128*4*4)
        self.de = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(32, out_ch, 4, 2, 1)
        )
        self.log_s = nn.Parameter(torch.tensor(-2.0))  # start with small scale
    def forward(self, z):
        mu = self.de(self.fc(z).view(z.size(0), 128, 4, 4))
        return mu, self.log_s

class VAE_DiscretizedLogistic(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, z_dim=32):
        super().__init__()
        self.enc = Encoder(in_ch, z_dim)
        self.dec = Decoder(out_ch, z_dim)
        self.delta = 1.0/256.0
    @staticmethod
    def reparam(mu, lv):
        std = (0.5*lv).exp(); eps = torch.randn_like(std); return mu + eps*std
    def log_discretized_logistic(self, x, mu, log_s):
        s = torch.exp(log_s)
        invs = 1.0 / s
        x_centered = x - mu
        plus = invs*(x_centered + 0.5*self.delta)
        minus = invs*(x_centered - 0.5*self.delta)
        cdf_plus = torch.sigmoid(plus)
        cdf_minus = torch.sigmoid(minus)
        probs = (cdf_plus - cdf_minus).clamp(min=1e-7)
        return torch.log(probs)
    def forward(self, x):
        mu_z, lv_z = self.enc(x)
        z = self.reparam(mu_z, lv_z)
        mu_x, log_s = self.dec(z)
        log_px = self.log_discretized_logistic(x, mu_x, log_s).flatten(1).sum(-1)
        kl = -0.5*(1 + lv_z - mu_z.pow(2) - lv_z.exp()).sum(1)
        return (-(log_px) + kl).mean()
