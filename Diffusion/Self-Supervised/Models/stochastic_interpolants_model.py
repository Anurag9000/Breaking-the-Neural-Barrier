# models/stochastic_interpolants_model.py
import torch
import torch.nn as nn
from models.flow_common_unet import VelocityUNet


class StochasticInterpolants(nn.Module):
    def __init__(self, base=64, channels=3):
        super().__init__()
        self.net = VelocityUNet(base=base, in_ch=channels, out_ch=channels)

    def sigma(self, t):
        # simple decreasing schedule σ(t)
        return 0.3 * (1 - t)

    def sigma_prime(self, t):
        return -0.3 * torch.ones_like(t)

    def loss(self, x0):
        B = x0.size(0)
        device = x0.device
        t = torch.rand(B, device=device)
        xi = torch.randn_like(x0)
        eps = torch.randn_like(x0)
        sig = self.sigma(t).view(-1, 1, 1, 1)
        x_t = (1 - t.view(-1, 1, 1, 1)) * xi + t.view(-1, 1, 1, 1) * x0 + sig * eps
        # time derivative of ψ_t: d/dt = (x0 - xi) + σ'(t) ε
        v_true = (x0 - xi) + self.sigma_prime(t).view(-1, 1, 1, 1) * eps
        v_hat = self.net(x_t, t)
        return (v_hat - v_true).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, steps=60, device='cuda', size=(3, 32, 32)):
        x = torch.randn(B, *size, device=device)
        t_grid = torch.linspace(0, 1, steps + 1, device=device)
        for i in range(steps):
            t = t_grid[i].repeat(B)
            dt = t_grid[i + 1] - t_grid[i]
            v = self.net(x, t)
            x = x + v * dt
        return x.clamp(-1, 1)
