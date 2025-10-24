# models/edm_model_min.py
import torch
import torch.nn as nn
from models.sde_common_unet import ScoreUNetSDE


# -------------------------------------------------------
# Minimal EDM (Elucidated Diffusion Model)
# -------------------------------------------------------
class EDM(nn.Module):
    def __init__(self, sigma_min=0.002, sigma_max=80.0, p_mean=-1.2, p_std=1.2, base=64, channels=3):
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.p_mean = p_mean
        self.p_std = p_std
        self.net = ScoreUNetSDE(base=base, in_ch=channels, out_ch=channels)

    # -------------------------------------------------------
    # Log-normal sigma sampling (EDM training noise schedule)
    # -------------------------------------------------------
    def _rand_sigma(self, B, device):
        log_sigma = self.p_mean + self.p_std * torch.randn(B, device=device)
        return torch.clamp(log_sigma.exp(), self.sigma_min, self.sigma_max)

    # -------------------------------------------------------
    # Denoising score matching loss (EDM form)
    # -------------------------------------------------------
    def loss(self, x):
        B = x.size(0)
        device = x.device
        sigma = self._rand_sigma(B, device)
        noise = torch.randn_like(x) * sigma.view(-1, 1, 1, 1)
        x_t = x + noise

        # EDM weighting and normalization
        c_in = 1 / torch.sqrt((sigma ** 2) + 1)
        c_out = sigma
        w = sigma ** 2 + 1

        # Network prediction: input is x_t, time normalized to [0,1]
        pred = self.net(x_t, (sigma - self.sigma_min) / (self.sigma_max - self.sigma_min))

        # True score under Gaussian noise
        true_score = -noise / (sigma.view(-1, 1, 1, 1) ** 2)
        base = (pred - true_score).pow(2).mean(dim=(1, 2, 3))
        return (w * base).mean()

    # -------------------------------------------------------
    # Euler Sampler (Karras et al. 2022)
    # -------------------------------------------------------
    @torch.no_grad()
    def sample_euler(self, B, steps=40, rho=7.0, device='cuda', size=(3, 32, 32)):
        """
        Karras et al. (2022) EDM sampler — Euler variant.
        """
        x = torch.randn(B, *size, device=device) * self.sigma_max

        sigmas = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(self.sigma_max)),
                torch.log(torch.tensor(self.sigma_min)),
                steps,
                device=device
            )
        )

        for i in range(steps - 1):
            s = sigmas[i].repeat(B)
            t = sigmas[i + 1].repeat(B)
            ds = (s - t)
            score = self.net(x, (s - self.sigma_min) / (self.sigma_max - self.sigma_min))
            d = -s.view(-1, 1, 1, 1) * score
            x = x + d * ds.view(-1, 1, 1, 1)

        return x.clamp(-1, 1)

    # -------------------------------------------------------
    # Heun’s Second-Order Sampler (Improved)
    # -------------------------------------------------------
    @torch.no_grad()
    def sample_heun(self, B, steps=40, device='cuda', size=(3, 32, 32)):
        """
        Heun's method: two-stage update for improved accuracy.
        """
        x = torch.randn(B, *size, device=device) * self.sigma_max

        sigmas = torch.exp(
            torch.linspace(
                torch.log(torch.tensor(self.sigma_max)),
                torch.log(torch.tensor(self.sigma_min)),
                steps,
                device=device
            )
        )

        for i in range(steps - 1):
            s = sigmas[i].repeat(B)
            t = sigmas[i + 1].repeat(B)
            ds = (t - s)

            # Stage 1: Euler step
            score_s = self.net(x, (s - self.sigma_min) / (self.sigma_max - self.sigma_min))
            d1 = -s.view(-1, 1, 1, 1) * score_s
            x_e = x + d1 * ds.view(-1, 1, 1, 1)

            # Stage 2: Evaluate at predicted next step
            score_t = self.net(x_e, (t - self.sigma_min) / (self.sigma_max - self.sigma_min))
            d2 = -t.view(-1, 1, 1, 1) * score_t

            # Combine updates
            x = x + 0.5 * (d1 + d2) * ds.view(-1, 1, 1, 1)

        return x.clamp(-1, 1)
