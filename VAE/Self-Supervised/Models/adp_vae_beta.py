from dataclasses import dataclass
from typing import Tuple, Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class BetaVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 32
    width: int = 128
    depth: int = 2
    beta: float = 4.0  # >1 for disentanglement


def feat_side(img_size: int, depth: int) -> int:
    return max(1, img_size // (2 ** depth))


class Encoder(nn.Module):
    def __init__(self, in_ch, width, depth, latent_dim):
        super().__init__()
        ch = width
        layers = [nn.Conv2d(in_ch, ch, 3, 2, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(ch, ch, 3, 2, 1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        self.net = nn.Sequential(*layers)
        fs = feat_side(32, depth)
        self.mu = nn.Linear(ch * fs * fs, latent_dim)
        self.logvar = nn.Linear(ch * fs * fs, latent_dim)

    def forward(self, x):
        h = self.net(x)
        h = torch.flatten(h, 1)
        return self.mu(h), self.logvar(h)


class Decoder(nn.Module):
    def __init__(self, out_ch, width, depth, latent_dim):
        super().__init__()
        fs = feat_side(32, depth)
        self.fc = nn.Linear(latent_dim, width * fs * fs)
        ups = []
        for _ in range(depth - 1):
            ups += [nn.ConvTranspose2d(width, width, 4, 2, 1), nn.BatchNorm2d(width), nn.ReLU(inplace=True)]
        ups += [nn.ConvTranspose2d(width, out_ch, 4, 2, 1)]
        self.ups = nn.Sequential(*ups)

    def forward(self, z):
        h = self.fc(z)
        fs = int((h.numel() // (h.size(0) * self.ups[0].in_channels)) ** 0.5)
        h = h.view(h.size(0), self.ups[0].in_channels, fs, fs)
        return torch.sigmoid(self.ups(h))


class BetaVAE(nn.Module):
    def __init__(self, cfg: BetaVAEConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(cfg.in_channels, cfg.width, cfg.depth, cfg.latent_dim)
        self.dec = Decoder(cfg.in_channels, cfg.width, cfg.depth, cfg.latent_dim)

    def encode(self, x):
        return self.enc(x)

    @staticmethod
    def reparameterize(mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.dec(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xr = self.decode(z)
        return xr, mu, logvar

    def loss_fn(self, x, xr, mu, logvar) -> Dict[str, torch.Tensor]:
        recon = F.binary_cross_entropy(xr, x, reduction='mean')
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        loss = recon + self.cfg.beta * kl
        return {"loss": loss, "recon": recon, "kl": kl}
