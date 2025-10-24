import torch
import torch.nn as nn
from models.common_unet import DenoiserUNet
from models.schedules import make_beta_schedule, DiffusionCoeffs


class DDPMX0(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) predicting x₀ (clean image)
    instead of ε (noise).
    """

    def __init__(self, img_size=32, channels=3, T=1000, schedule="linear", base=64):
        super().__init__()
        self.T = T
        self.channels = channels
        self.net = DenoiserUNet(base=base, in_ch=channels, out_ch=channels)

        betas = make_beta_schedule(T, schedule)
        self.register_buffer("betas", betas)
        self.coeff = DiffusionCoeffs(betas)

    # ----------------------------------------------------------
    # q(x_t | x_0): forward diffusion process
    # ----------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self.coeff.sqrt_alpha_bars[t][:, None, None, None] * x0
            + self.coeff.sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
        )

    # ----------------------------------------------------------
    # Training step: predict x₀ (the clean image)
    # ----------------------------------------------------------
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        x0_hat = self.net(x_t, t)
        return nn.functional.mse_loss(x0_hat, x0)

    # ----------------------------------------------------------
    # Convert predicted x₀ → predicted ε
    # ----------------------------------------------------------
    @torch.no_grad()
    def predict_eps(self, x_t: torch.Tensor, t: torch.Tensor, x0_hat: torch.Tensor) -> torch.Tensor:
        c = self.coeff
        return (
            (x_t - c.sqrt_alpha_bars[t][:, None, None, None] * x0_hat)
            / c.sqrt_one_minus_alpha_bars[t][:, None, None, None]
        )

    # ----------------------------------------------------------
    # Reverse diffusion: p(x_{t-1} | x_t)
    # ----------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        x0_hat = self.net(x_t, t)
        eps = self.predict_eps(x_t, t, x0_hat)
        c = self.coeff

        mean = (
            c.posterior_mean_coef1[t][:, None, None, None] * x_t
            + c.posterior_mean_coef2[t][:, None, None, None] * x0_hat
        )

        var = c.posterior_variance[t][:, None, None, None]

        # If at t=0, don't add noise
        if (t == 0).all():
            return mean

        noise = torch.randn_like(x_t)
        return mean + torch.sqrt(var) * noise

    # ----------------------------------------------------------
    # Full sampling loop
    # ----------------------------------------------------------
    @torch.no_grad()
    def sample(self, B: int, device: str = "cuda") -> torch.Tensor:
        x = torch.randn(B, self.channels, 32, 32, device=device)
        for step in reversed(range(self.T)):
            t = torch.full((B,), step, device=device, dtype=torch.long)
            x = self.p_sample(x, t)
        return x.clamp(-1, 1)
