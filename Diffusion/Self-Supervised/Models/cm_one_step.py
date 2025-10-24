# models/cm_one_step.py
import torch
import torch.nn as nn
from models.g_common_cunet import ConsistencyUNet


class ZECM(nn.Module):
    """
    Zero-Error Consistency Model (ZECM) for single-step denoising.

    Supports both Variance-Preserving (VP) and Variance-Exploding (VE) modes.
    """
    def __init__(self, mode='vp', T=1000, sigma_min=0.01, sigma_max=50.0, base=64, channels=3):
        super().__init__()
        assert mode in ['vp', 've'], "mode must be 'vp' or 've'"
        self.mode = mode
        self.T = T
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

        self.net = ConsistencyUNet(base=base, in_ch=channels, out_ch=channels)

        if mode == 'vp':
            # Linear beta schedule
            betas = torch.linspace(1e-4, 2e-2, T)
            self.register_buffer('alpha_bars', torch.cumprod(1.0 - betas, dim=0))

    def _vp_q(self, x0, t, eps):
        """VP forward diffusion: x_t = sqrt(alpha_bar) * x0 + sqrt(1-alpha_bar) * eps"""
        ab = self.alpha_bars[t].view(-1, 1, 1, 1)
        return torch.sqrt(ab) * x0 + torch.sqrt(1 - ab) * eps

    def _ve_sigma(self, t01):
        """VE noise schedule: sigma(t) = sigma_min * (sigma_max/sigma_min)^t"""
        return self.sigma_min * (self.sigma_max / self.sigma_min) ** t01

    def _ve_q(self, x0, t01, eps):
        """VE forward diffusion: x_t = x0 + sigma(t) * eps"""
        sig = self._ve_sigma(t01).view(-1, 1, 1, 1)
        return x0 + sig * eps

    def loss(self, x0):
        """
        Compute single-step consistency loss.
        VP: predict x0 from noisy x_t
        VE: predict x0 from noisy x_t
        """
        B = x0.size(0)
        device = x0.device

        if self.mode == 'vp':
            t = torch.randint(1, self.T, (B,), device=device)
            x_t = self._vp_q(x0, t, torch.randn_like(x0))
            x_hat = self.net(x_t, t.float() / self.T, torch.zeros_like(t, dtype=torch.float32))
        else:  # VE
            t = torch.rand(B, device=device)
            x_t = self._ve_q(x0, t, torch.randn_like(x0))
            x_hat = self.net(x_t, t, torch.zeros_like(t))

        return (x_hat - x0).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, device='cuda', size=(3, 32, 32), t_start=1.0):
        """
        Generate a single-step sample from random noise.
        """
        x = torch.randn(B, *size, device=device)
        t = torch.full((B,), t_start, device=device)
        s = torch.zeros_like(t)
        return self.net(x, t, s).clamp(-1, 1)
