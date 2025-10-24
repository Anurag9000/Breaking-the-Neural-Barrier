# models/dsm_model.py
import torch
import torch.nn as nn
from models.sm_common_unet import ScoreUNet


class DSM(nn.Module):
    """Denoising Score Matching (DSM) model with fixed Gaussian noise."""

    def __init__(self, sigma=0.2, base=64, channels=3):
        super().__init__()
        self.sigma = sigma
        self.net = ScoreUNet(base=base, in_ch=channels, out_ch=channels)

    def loss(self, x):
        """
        Compute denoising score matching loss:
        L = E[ || s_theta(x + noise) + noise / sigma^2 ||^2 ]
        """
        B = x.size(0)
        device = x.device
        sigma = torch.full((B,), self.sigma, device=device)
        noise = torch.randn_like(x) * self.sigma
        x_t = x + noise

        # Score target for Gaussian noise
        target = -noise / (self.sigma ** 2)

        # Predicted score
        pred = self.net(x_t, sigma)
        return (pred - target).pow(2).mean()

    @torch.no_grad()
    def langevin_sample(self, B, steps=200, step_size=0.01, device='cuda', size=(3, 32, 32)):
        """
        Sample from the model using Annealed Langevin Dynamics.
        """
        x = torch.randn(B, *size, device=device)
        sigma = torch.full((B,), self.sigma, device=device)

        for _ in range(steps):
            grad = self.net(x, sigma)
            x = x + 0.5 * step_size * grad + (step_size ** 0.5) * torch.randn_like(x)

        return x.clamp(-1, 1)
