# models/cm_ve.py
import torch
import torch.nn as nn
from models.g_common_cunet import ConsistencyUNet


class VEConsistency(nn.Module):
    """
    Variance Exploding (VE) consistency model.
    Predicts x_s given x_t for s < t using VE-type diffusion.
    """
    def __init__(self, sigma_min=0.01, sigma_max=50.0, base=64, channels=3):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.net = ConsistencyUNet(base=base, in_ch=channels, out_ch=channels)

    def sigma(self, t01):
        """
        Compute sigma(t) for VE schedule.
        t01: tensor in [0,1]
        """
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t01

    def q_sample(self, x0, t01, eps):
        """
        Forward diffusion: x_t = x0 + sigma(t) * eps
        """
        sig = self.sigma(t01).view(-1, 1, 1, 1)
        return x0 + sig * eps

    def loss(self, x0):
        """
        Train the consistency model to predict x_s from x_t.
        """
        B = x0.size(0)
        device = x0.device

        t = torch.rand(B, device=device)
        s = torch.rand(B, device=device)

        # Enforce s <= t
        swap = s > t
        s, t = torch.where(swap, t, s), torch.where(swap, s, t)

        eps_t = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, eps_t)

        eps_s = torch.randn_like(x0)
        x_s = self.q_sample(x0, s, eps_s)  # independent noise for x_s

        x_hat = self.net(x_t, t, s)
        return (x_hat - x_s).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, steps=40, device='cuda', size=(3, 32, 32)):
        """
        Iterative sampling using the learned consistency function.
        """
        x = torch.randn(B, *size, device=device) * self.sigma_max
        ts = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t = ts[i].repeat(B)
            s = ts[i + 1].repeat(B)
            x = self.net(x, t, s)

        return x.clamp(-1, 1)
