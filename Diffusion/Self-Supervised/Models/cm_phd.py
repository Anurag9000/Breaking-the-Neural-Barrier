# models/cm_phd.py
import torch
import torch.nn as nn
from models.g_common_cunet import ConsistencyUNet


class ProgressiveHalving(nn.Module):
    """
    Progressive Halving Consistency Model (PhD) for multi-step denoising.

    Supports VP (Variance-Preserving) and VE (Variance-Exploding) modes.
    Iteratively maps x_t -> x_{t/2} (VP) or x_t -> x_{t*0.5} (VE) in rounds.
    """
    def __init__(self, mode='vp', T=1000, base=64, channels=3, sigma_min=0.01, sigma_max=50.0):
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

    def _half(self, t):
        """Compute halved time: integer division for VP, 0.5x for VE"""
        return (t // 2) if self.mode == 'vp' else t * 0.5

    def loss(self, x0):
        """
        Progressive halving loss: supervise network to map x_t -> x_{t/2}.
        """
        B = x0.size(0)
        device = x0.device

        if self.mode == 'vp':
            t = torch.randint(2, self.T, (B,), device=device)
            x_t = self._vp_q(x0, t, torch.randn_like(x0))
            s = t // 2
            x_s = self._vp_q(x0, s, torch.randn_like(x0))
            x_hat = self.net(x_t, t.float() / self.T, s.float() / self.T)
        else:  # VE
            t = torch.rand(B, device=device)
            x_t = self._ve_q(x0, t, torch.randn_like(x0))
            s = t * 0.5
            x_s = self._ve_q(x0, s, torch.randn_like(x0))
            x_hat = self.net(x_t, t, s)

        return (x_hat - x_s).pow(2).mean()

    @torch.no_grad()
    def sample(self, B, rounds=5, device='cuda', size=(3, 32, 32)):
        """
        Generate samples using progressive halving over several rounds.
        """
        x = torch.randn(B, *size, device=device)

        if self.mode == 'vp':
            t = torch.full((B,), float(self.T - 1), device=device)
            for _ in range(rounds):
                s = (t // 2).float()
                x = self.net(x, t / self.T, s / self.T)
                t = t // 2
        else:  # VE
            t = torch.ones(B, device=device)
            for _ in range(rounds):
                s = t * 0.5
                x = self.net(x, t, s)
                t = s

        return x.clamp(-1, 1)
