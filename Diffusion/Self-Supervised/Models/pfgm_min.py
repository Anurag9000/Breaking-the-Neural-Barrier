# models/pfgm_min.py
import torch
import torch.nn as nn
from models.e_common_unet import UNetE


class PFGM(nn.Module):
    def __init__(self, base=64, channels=3, eps=1e-3):
        super().__init__()
        self.net = UNetE(base=base, in_ch=channels, out_ch=channels)
        self.eps = eps

    def loss(self, x0):
        """
        PFGM loss based on weighted squared error of vector field
        """
        B = x0.size(0)
        device = x0.device

        # interpolation time and random initial points
        t = torch.rand(B, device=device)
        xi = torch.randn_like(x0)
        x_t = (1 - t.view(-1, 1, 1, 1)) * xi + t.view(-1, 1, 1, 1) * x0

        # distance squared + epsilon for stability
        r2 = (x_t - x0).pow(2).mean(dim=(1, 2, 3)) + self.eps

        # true vector field and predicted vector field
        v_true = x0 - xi
        v_hat = self.net(x_t, t)

        # inverse-square weighting
        w = 1.0 / r2
        l = (v_hat - v_true).pow(2).mean(dim=(1, 2, 3)) * w
        return l.mean()

    @torch.no_grad()
    def sample(self, B, steps=50, device='cuda', size=(3, 32, 32)):
        """
        Generate samples from the PFGM model
        """
        x = torch.randn(B, *size, device=device)
        t_grid = torch.linspace(0, 1, steps + 1, device=device)

        for i in range(steps):
            t = t_grid[i].repeat(B)
            dt = t_grid[i + 1] - t_grid[i]
            v = self.net(x, t)
            x = x + v * dt

        return x.clamp(-1, 1)
