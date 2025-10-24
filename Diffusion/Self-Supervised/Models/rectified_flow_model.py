# models/rectified_flow_model.py
import torch
import torch.nn as nn
from models.flow_common_unet import VelocityUNet


class RectifiedFlow(nn.Module):
    """Rectified Flow model implementing velocity-based generative dynamics."""

    def __init__(self, base=64, channels=3):
        super().__init__()
        self.net = VelocityUNet(base=base, in_ch=channels, out_ch=channels)

    def loss(self, x0):
        """Compute velocity matching loss for rectified flow training."""
        B = x0.size(0)
        device = x0.device

        # Sample random time t ∈ [0,1]
        t = torch.rand(B, device=device)
        xi = torch.randn_like(x0)

        # Interpolate between noise and data
        x_t = (1 - t.view(-1, 1, 1, 1)) * xi + t.view(-1, 1, 1, 1) * x0

        # True velocity: derivative of straight-line path
        v_true = x0 - xi

        # Predicted velocity
        v_hat = self.net(x_t, t)

        # Mean squared error between predicted and true velocities
        return (v_hat - v_true).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, steps=50, device='cuda', size=(3, 32, 32)):
        """
        Sample images using Euler integration over the learned flow field:
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
