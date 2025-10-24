# models/laplace_diffusion_model.py
import torch
import torch.nn as nn
from models.e_common_unet import UNetE


class LaplaceDiffusion(nn.Module):
    def __init__(self, T=1000, base=64, channels=3, b_min=1e-3, b_max=0.5):
        super().__init__()
        self.T = T
        self.b_min = b_min
        self.b_max = b_max
        self.net = UNetE(base=base, in_ch=channels, out_ch=channels)

    def b(self, t):
        """
        Scale schedule between b_min and b_max
        """
        return self.b_min + (self.b_max - self.b_min) * (t.float() / self.T)

    def q_sample(self, x0, t, noise):
        """
        Variance-preserving style blend for Laplace diffusion
        """
        b = self.b(t).view(-1, 1, 1, 1)
        alpha = torch.exp(-b)  # pseudo-alpha from Laplace scale
        return alpha * x0 + (1 - alpha) * noise

    def laplace_noise(self, x, scale):
        """
        Sample Laplace noise
        """
        u = torch.rand_like(x) - 0.5
        return scale * torch.sign(u) * torch.log1p(-2 * torch.abs(u) + 1e-8)

    def loss(self, x0):
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device)
        noise = self.laplace_noise(x0, 1.0)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = self.net(x_t, t)
        return (eps_hat - noise).abs().mean()  # L1 loss fits Laplace

    @torch.no_grad()
    def sample(self, B, steps=100, device='cuda', size=(3, 32, 32)):
        x = torch.randn(B, *size, device=device)
        for s in reversed(range(steps)):
            t = torch.full((B,), int(self.T * s / steps), device=device, dtype=torch.long)
            eps = self.net(x, t)
            alpha = torch.exp(-self.b(t)).view(-1, 1, 1, 1)
            x0_hat = (x - (1 - alpha) * eps) / (alpha + 1e-8)
            # Predictor step toward x0_hat
            x = alpha * x + (1 - alpha) * x0_hat
            # Corrector (Laplace perturb)
            noise = self.laplace_noise(x, 0.01)
            x = x + noise
        return x.clamp(-1, 1)
