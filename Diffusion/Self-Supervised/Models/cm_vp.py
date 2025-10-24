# models/cm_vp.py
import math
import torch
import torch.nn as nn
from models.g_common_cunet import ConsistencyUNet


class VPConsistency(nn.Module):
    """
    Variance Preserving (VP) consistency model for continuous-time diffusion.
    Predicts x_s given x_t for s < t.
    """
    def __init__(self, T=1000, schedule='linear', beta_start=1e-4, beta_end=2e-2, base=64, channels=3):
        super().__init__()
        self.T = T
        self.channels = channels
        self.net = ConsistencyUNet(base=base, in_ch=channels, out_ch=channels)

        # Build beta schedule
        if schedule == 'linear':
            betas = torch.linspace(beta_start, beta_end, T)
        elif schedule == 'cosine':
            t = torch.linspace(0, 1, T + 1, dtype=torch.float64)
            s = 0.008
            f = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
            alpha_bar = f / f[0]
            betas = (1 - alpha_bar[1:] / alpha_bar[:-1]).to(torch.float32).clamp(1e-8, 0.999)
        else:
            raise ValueError(f'Unknown schedule: {schedule}')

        self.register_buffer('betas', betas)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        self.register_buffer('alpha_bars', alpha_bars)

    def _alpha_bar(self, t):
        """
        Returns cumulative product of alphas at integer timestep t.
        t: tensor of shape (B,) with values in [0, T-1]
        """
        t = t.clamp(0, self.T - 1)
        return self.alpha_bars[t]

    def q_sample(self, x0, t, eps):
        """
        Diffusion forward process: q(x_t | x0)
        x0: (B,C,H,W)
        t: tensor of timesteps
        eps: noise
        """
        ab = self._alpha_bar(t).view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps

    def loss(self, x0):
        """
        Train consistency model to predict x_s from x_t
        using the same underlying noise.
        """
        B = x0.size(0)
        device = x0.device

        # Sample random timesteps
        t = torch.randint(1, self.T, (B,), device=device)
        s = torch.randint(0, self.T, (B,), device=device)

        # Ensure s < t
        swap = s > t
        s, t = torch.where(swap, t, s), torch.where(swap, s, t)

        eps = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps)
        x_s = self.q_sample(x0, s, eps=torch.randn_like(x0))  # independent noise for x_s

        x_hat = self.net(x_t, t.float() / self.T, s.float() / self.T)
        return (x_hat - x_s).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, steps=40, device='cuda', size=(3, 32, 32)):
        """
        Iterative sampling using the learned consistency function.
        """
        x = torch.randn(B, *size, device=device)
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)  # monotone time grid

        for i in range(steps):
            t = ts[i].repeat(B)
            s = ts[i + 1].repeat(B)
            x = self.net(x, t, s)

        return x.clamp(-1, 1)
