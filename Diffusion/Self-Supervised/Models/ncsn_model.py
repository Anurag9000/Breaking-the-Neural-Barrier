# models/ncsn_model.py
import torch
import torch.nn as nn
from models.sm_common_unet import ScoreUNet


class NCSN(nn.Module):
    """Noise-Conditional Score Network (NCSN) with multiple noise levels."""

    def __init__(self, sigmas=None, base=64, channels=3):
        super().__init__()
        if sigmas is None:
            # Geometric ladder from high to low noise
            self.sigmas = torch.exp(
                torch.linspace(torch.log(torch.tensor(1.0)), torch.log(torch.tensor(0.01)), 10)
            )
        else:
            self.sigmas = torch.tensor(sigmas, dtype=torch.float32)

        self.net = ScoreUNet(base=base, in_ch=channels, out_ch=channels)

    def loss(self, x):
        """
        DSM loss over multiple noise scales:
        L = E[ || s_theta(x + noise) + noise / sigma^2 ||^2 ]
        """
        B = x.size(0)
        device = x.device

        # Sample a sigma for each example
        idx = torch.randint(0, len(self.sigmas), (B,), device=device)
        sigma = self.sigmas[idx].to(device)
        noise = torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
        x_t = x + noise

        # Score target for Gaussian noise
        target = -noise / (sigma.view(-1, 1, 1, 1) ** 2)

        pred = self.net(x_t, sigma)
        # Can weight by sigma^2 per DSM paper; using 1 here for stability
        return (pred - target).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, steps_per_level=100, device='cuda', size=(3, 32, 32)):
        """
        Generate samples using annealed Langevin dynamics.
        """
        x = torch.randn(B, *size, device=device)

        for sigma in self.sigmas.to(device):
            s = torch.full((B,), float(sigma.item()), device=device)
            step = (sigma ** 2) * 0.1  # conservative step scale

            for _ in range(steps_per_level):
                grad = self.net(x, s)
                x = x + 0.5 * step * grad + (step ** 0.5) * torch.randn_like(x)

        return x.clamp(-1, 1)
