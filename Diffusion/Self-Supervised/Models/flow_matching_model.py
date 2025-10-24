# models/flow_matching_model.py
import torch
import torch.nn as nn
from models.flow_common_unet import VelocityUNet


class FlowMatching(nn.Module):
    """Flow Matching model with optional weighting schemes."""

    def __init__(self, base=64, channels=3, weight='uniform'):
        """
        Args:
            base (int): Base channel multiplier.
            channels (int): Number of input/output image channels.
            weight (str): Weighting strategy for loss ('uniform' or 'speed').
        """
        super().__init__()
        self.net = VelocityUNet(base=base, in_ch=channels, out_ch=channels)
        self.weight = weight

    def loss(self, x0):
        """Compute flow matching loss."""
        B = x0.size(0)
        device = x0.device

        # Sample random time t ∈ [0,1]
        t = torch.rand(B, device=device)
        xi = torch.randn_like(x0)

        # Linear interpolation between noise and data
        x_t = (1 - t.view(-1, 1, 1, 1)) * xi + t.view(-1, 1, 1, 1) * x0

        # True velocity field (x0 - xi)
        v_true = x0 - xi

        # Predicted velocity from network
        v_hat = self.net(x_t, t)

        # Base loss (mean squared velocity error)
        l = (v_hat - v_true).pow(2).mean(dim=(1, 2, 3))

        # Optional weighting by speed magnitude
        if self.weight == 'speed':
            w = v_true.pow(2).mean(dim=(1, 2, 3)) + 1e-6
            l = l * w

        return l.mean()

    @torch.no_grad()
    def sample(self, B, steps=50, device='cuda', size=(3, 32, 32)):
        """
        Generate samples via Euler integration of the learned flow field.
        dx/dt = v_theta(x, t)
        """
        x = torch.randn(B, *size, device=device)
        t_grid = torch.linspace(0, 1, steps + 1, device=device)

        for i in range(steps):
            t = t_grid[i].repeat(B)
            dt = t_grid[i + 1] - t_grid[i]
            v = self.net(x, t)
            x = x + v * dt

        return x.clamp(-1, 1)
