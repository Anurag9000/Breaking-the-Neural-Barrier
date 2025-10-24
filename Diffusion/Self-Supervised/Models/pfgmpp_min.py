# models/pfgmpp_min.py
import torch
import torch.nn as nn
from models.e_common_unet import UNetE


class PFGMpp(nn.Module):
    def __init__(self, base=64, channels=3, r_min=0.1, r_max=10.0):
        super().__init__()
        self.net = UNetE(base=base, in_ch=channels, out_ch=channels)
        self.r_min = r_min
        self.r_max = r_max

    def radius(self, t):
        """
        Geometric interpolation of radius between r_min and r_max
        """
        return self.r_min * (self.r_max / self.r_min) ** t

    def loss(self, x0):
        """
        PFGM++ weighted loss using radius normalization
        """
        B = x0.size(0)
        device = x0.device

        # random interpolation time and base points
        t = torch.rand(B, device=device)
        xi = torch.randn_like(x0)
        r = self.radius(t).view(-1, 1, 1, 1)

        # interpolated positions
        x_t = (1 - t.view(-1, 1, 1, 1)) * r * xi + t.view(-1, 1, 1, 1) * x0

        # true vector field and prediction
        v_true = x0 - r * xi
        v_hat = self.net(x_t, t)

        # normalize by radius magnitude
        w = 1.0 / (r.abs().mean(dim=(1, 2, 3)) + 1e-6)
        l = (v_hat - v_true).pow(2).mean(dim=(1, 2, 3)) * w

        return l.mean()

    @torch.no_grad()
    def sample(self, B, steps=50, device='cuda', size=(3, 32, 32)):
        """
        Generate samples from PFGM++ model
        """
        x = torch.randn(B, *size, device=device) * self.r_max
        t_grid = torch.linspace(0, 1, steps + 1, device=device)

        for i in range(steps):
            t = t_grid[i].repeat(B)
            dt = t_grid[i + 1] - t_grid[i]
            v = self.net(x, t)
            x = x + v * dt

        return x.clamp(-1, 1)
