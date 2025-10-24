# models/score_sde_unified.py
import torch
import torch.nn as nn
from models.sde_common_unet import ScoreUNetSDE


# ----------------------------
# Unified Score SDE Model
# ----------------------------
class ScoreSDEUnified(nn.Module):
    def __init__(self, sde_type='ve', sigma_min=0.01, sigma_max=50.0, T=1.0, base=64, channels=3):
        super().__init__()
        assert sde_type in ['ve', 'vp', 'subvp'], "sde_type must be one of ['ve', 'vp', 'subvp']"
        self.type = sde_type
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.T = T
        self.net = ScoreUNetSDE(base=base, in_ch=channels, out_ch=channels)

    # -------------------------------------------------------
    # Noise scale σ(t) for different SDE variants
    # -------------------------------------------------------
    def _sigma(self, t):
        if self.type == 've':
            # Geometric schedule: σ(t) = σ_min * (σ_max / σ_min)^t
            return self.sigma_min * (self.sigma_max / self.sigma_min) ** t
        else:
            # VP / subVP use a beta(t) schedule (log-SNR parameterization)
            beta0, beta1 = 0.1, 20.0
            beta_t = beta0 + t * (beta1 - beta0)
            # Approximate cumulative variance integral
            return torch.sqrt(1 - torch.exp(-2 * beta_t * t))

    # -------------------------------------------------------
    # Denoising score matching loss
    # -------------------------------------------------------
    def loss(self, x):
        B = x.size(0)
        device = x.device
        t = torch.rand(B, device=device)
        sigma = self._sigma(t).view(-1, 1, 1, 1)
        noise = torch.randn_like(x) * sigma

        if self.type == 've':
            x_t = x + noise
        else:
            x_t = torch.sqrt(1 - sigma ** 2) * x + noise  # VP/subVP corruption

        # True score of corrupted x under Gaussian noise
        if self.type == 've':
            true_score = -noise / (sigma ** 2)
        else:
            true_score = ((torch.sqrt(1 - sigma ** 2) * x - x_t) / (sigma ** 2 + 1e-8))

        pred = self.net(x_t, t)
        return (pred - true_score).pow(2).mean()

    # -------------------------------------------------------
    # SDE drift and diffusion functions
    # -------------------------------------------------------
    def sde(self, x, t):
        if self.type == 've':
            g = self._sigma(t) * ((self.sigma_max / self.sigma_min) ** (t * 0))
            drift = torch.zeros_like(x)
            diffusion = g

        elif self.type == 'vp':
            beta0, beta1 = 0.1, 20.0
            beta_t = beta0 + t * (beta1 - beta0)
            drift = -0.5 * beta_t.view(-1, 1, 1, 1) * x
            diffusion = torch.sqrt(beta_t)

        else:  # subVP
            beta0, beta1 = 0.1, 20.0
            beta_t = beta0 + t * (beta1 - beta0)
            mean_coef = torch.exp(-0.5 * beta_t * t)
            drift = -0.5 * beta_t.view(-1, 1, 1, 1) * (x - mean_coef.view(-1, 1, 1, 1) * x)
            diffusion = torch.sqrt(beta_t)

        return drift, diffusion

    # -------------------------------------------------------
    # Sampling via reverse SDE
    # -------------------------------------------------------
    @torch.no_grad()
    def sample_sde(self, B, steps=1000, device='cuda', size=(3, 32, 32)):
        x = torch.randn(B, *size, device=device)
        t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t = t_grid[i].repeat(B)
            dt = (t_grid[i + 1] - t_grid[i])
            drift, diff = self.sde(x, t)
            score = self.net(x, t)
            x = (
                x
                + (drift - (diff.view(-1, 1, 1, 1) ** 2) * score) * dt
                + torch.sqrt(torch.abs(dt))
                * diff.view(-1, 1, 1, 1)
                * torch.randn_like(x)
            )
        return x.clamp(-1, 1)

    # -------------------------------------------------------
    # Sampling via Probability-Flow ODE (deterministic)
    # -------------------------------------------------------
    @torch.no_grad()
    def sample_ode(self, B, steps=1000, device='cuda', size=(3, 32, 32)):
        x = torch.randn(B, *size, device=device)
        t_grid = torch.linspace(1.0, 0.0, steps + 1, device=device)

        for i in range(steps):
            t = t_grid[i].repeat(B)
            dt = (t_grid[i + 1] - t_grid[i])
            _, diff = self.sde(x, t)
            score = self.net(x, t)
            x = x - 0.5 * (diff.view(-1, 1, 1, 1) ** 2) * score * dt

        return x.clamp(-1, 1)
