# models/conditional_flow_matching_model.py
import torch
import torch.nn as nn
from models.flow_common_unet import VelocityUNet


class ConditionalFlowMatching(nn.Module):
    """Conditional Flow Matching model with optional null-token guidance."""

    def __init__(self, base=64, channels=3, p_null=0.1):
        """
        Args:
            base (int): Base channel multiplier.
            channels (int): Number of input/output image channels.
            p_null (float): Probability of using null token (classifier-free guidance).
        """
        super().__init__()
        self.p_null = p_null
        self.net = VelocityUNet(base=base, in_ch=channels, out_ch=channels, use_null_token=True)

    def loss(self, x0):
        """Compute conditional flow matching loss."""
        B = x0.size(0)
        device = x0.device

        # Random time and noise
        t = torch.rand(B, device=device)
        cond_noise = torch.randn_like(x0)

        # Linear interpolation between noise and data
        x_t = (1 - t.view(-1, 1, 1, 1)) * cond_noise + t.view(-1, 1, 1, 1) * x0

        # True velocity
        v_true = x0 - cond_noise

        # Determine which samples use null token
        use_null = (torch.rand(B, device=device) < self.p_null)

        # Predicted velocity
        v_hat = self.net(x_t, t, use_null=use_null)

        return (v_hat - v_true).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, steps=50, device='cuda', size=(3, 32, 32), guidance=0.0):
        """
        Generate samples with optional classifier-free-style guidance.

        Args:
            B (int): Batch size.
            steps (int): Number of integration steps.
            device (str): Device to run on.
            size (tuple): Image size (C, H, W).
            guidance (float): Classifier-free guidance strength.
        """
        x = torch.randn(B, *size, device=device)
        t_grid = torch.linspace(0, 1, steps + 1, device=device)

        for i in range(steps):
            t = t_grid[i].repeat(B)
            dt = t_grid[i + 1] - t_grid[i]

            # Classifier-free guidance with null token
            v_null = self.net(x, t, use_null=True)
            v_cond = self.net(x, t, use_null=False)
            v = (1 + guidance) * v_cond - guidance * v_null

            x = x + v * dt

        return x.clamp(-1, 1)
