import math
from dataclasses import dataclass
from typing import Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class VAEConfig:
    in_channels: int = 3
    img_size: int = 32
    latent_dim: int = 64
    width: int = 128
    depth: int = 2


class ConvEncoder(nn.Module):
    def __init__(self, in_ch: int, width: int, depth: int, latent_dim: int):
        super().__init__()
        layers = []
        ch = width
        layers += [nn.Conv2d(in_ch, ch, 3, stride=2, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        for _ in range(depth - 1):
            layers += [nn.Conv2d(ch, ch, 3, stride=2, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        self.feature = nn.Sequential(*layers)
        self.to_mu = nn.Linear(ch * (img_feat_side(img_size=32, depth=depth) ** 2), latent_dim)
        self.to_logvar = nn.Linear(ch * (img_feat_side(img_size=32, depth=depth) ** 2), latent_dim)
        self.depth = depth
        self.ch = ch

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.feature(x)
        h = torch.flatten(h, 1)
        return self.to_mu(h), self.to_logvar(h)


class ConvDecoder(nn.Module):
    def __init__(self, out_ch: int, width: int, depth: int, latent_dim: int):
        super().__init__()
        feat_side = img_feat_side(img_size=32, depth=depth)
        self.fc = nn.Linear(latent_dim, width * feat_side * feat_side)
        ups = []
        ch = width
        for i in range(depth - 1):
            ups += [nn.ConvTranspose2d(ch, ch, 4, stride=2, padding=1), nn.BatchNorm2d(ch), nn.ReLU(inplace=True)]
        ups += [nn.ConvTranspose2d(ch, out_ch, 4, stride=2, padding=1)]
        self.ups = nn.Sequential(*ups)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        # assuming square feature maps
        side = int(math.sqrt(h.shape[1] // self.ups[0].in_channels)) if isinstance(self.ups[0], nn.ConvTranspose2d) else 4
        h = h.view(h.size(0), -1, side, side)
        x_recon = self.ups(h)
        return torch.sigmoid(x_recon)


def img_feat_side(img_size: int, depth: int) -> int:
    # each encoder layer halves spatial dims; start with stride-2 convs depth times
    return max(1, img_size // (2 ** depth))


class VanillaVAE(nn.Module):
    def __init__(self, cfg: VAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ConvEncoder(cfg.in_channels, cfg.width, cfg.depth, cfg.latent_dim)
        self.decoder = ConvDecoder(cfg.in_channels, cfg.width, cfg.depth, cfg.latent_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoder(x)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decode(z)
        return x_recon, mu, logvar

    @staticmethod
    def loss_fn(x: torch.Tensor, x_recon: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> Dict[str, torch.Tensor]:
        # Reconstruction: BCE over pixels
        recon_loss = F.binary_cross_entropy(x_recon, x, reduction='mean')
        # KL divergence to N(0, I)
        kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()
        elbo = recon_loss + kl
        return {"loss": elbo, "recon": recon_loss, "kl": kl}
