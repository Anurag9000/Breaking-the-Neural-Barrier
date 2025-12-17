import torch
import torch.nn as nn
from typing import Tuple


class ConvEncoder(nn.Module):
    def __init__(self, in_channels: int, width: int, depth: int, latent_dim: int):
        super().__init__()
        layers = []
        ch = width
        layers.append(nn.Conv2d(in_channels, ch, kernel_size=3, padding=1))
        layers.append(nn.ReLU(inplace=True))
        for i in range(1, depth):
            layers.append(nn.Conv2d(ch, ch, kernel_size=3, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.net = nn.Sequential(*layers)
        self.latent_dim = latent_dim
        self.width = width
        self.depth = depth
        # The spatial size after depth-1 strided convolutions (approx; assumes 32x32 input)
        self.spatial = 32 // (2 ** max(depth - 1, 0))
        feat_dim = ch * self.spatial * self.spatial
        self.fc_mu = nn.Linear(feat_dim, latent_dim)
        self.fc_logvar = nn.Linear(feat_dim, latent_dim)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        h_flat = h.view(h.size(0), -1)
        mu = self.fc_mu(h_flat)
        logvar = self.fc_logvar(h_flat)
        return mu, logvar


class ConvDecoder(nn.Module):
    def __init__(self, out_channels: int, width: int, depth: int, latent_dim: int):
        super().__init__()
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.spatial = 32 // (2 ** max(depth - 1, 0))
        feat_dim = width * self.spatial * self.spatial
        self.fc = nn.Linear(latent_dim, feat_dim)

        layers = []
        ch = width
        for i in range(depth - 1, 0, -1):
            # upsample by factor 2 using convtranspose
            layers.append(nn.ConvTranspose2d(ch, ch, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
        self.deconv = nn.Sequential(*layers)
        self.out = nn.Conv2d(ch, out_channels, kernel_size=3, padding=1)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(z.size(0), self.width, self.spatial, self.spatial)
        h = self.deconv(h)
        x_rec = self.out(h)
        return x_rec


class DAEVAEConv(nn.Module):
    """
    Variational Conv DAE (denoising VAE) for CIFAR images.

    - width: base channel count.
    - depth: number of conv/convtranspose blocks.
    - latent_dim: dimensionality of latent z.
    """

    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 3, latent_dim: int = 128):
        super().__init__()
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.latent_dim = latent_dim
        self.encoder = ConvEncoder(in_channels, width, depth, latent_dim)
        self.decoder = ConvDecoder(in_channels, width, depth, latent_dim)

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
        x_rec = self.decode(z)
        return x_rec, mu, logvar


def dae_vae_total_neurons(width: int, depth: int, latent_dim: int) -> int:
    # simple capacity proxy
    return int(width * depth * 32 + latent_dim)

