import torch
import torch.nn as nn
from models.common_unet import DenoiserUNet
from models.schedules import make_beta_schedule, DiffusionCoeffs


class DDPMEps(nn.Module):
    """
    Denoising Diffusion Probabilistic Model (DDPM) predicting epsilon (noise).
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
    # Training step: predict epsilon (noise)
    # ----------------------------------------------------------
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        device = x0.device
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = self.net(x_t, t)
        return nn.functional.mse_loss(eps_hat, noise)

    # ----------------------------------------------------------
    # Reverse diffusion: p(x_{t-1} | x_t)
    # ----------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps = self.net(x_t, t)
        coef = self.coeff

        mean = (
            coef.posterior_mean_coef1[t][:, None, None, None] * x_t
            + coef.posterior_mean_coef2[t][:, None, None, None]
            * (
                (x_t - coef.sqrt_one_minus_alpha_bars[t][:, None, None, None] * eps)
                / coef.sqrt_alpha_bars[t][:, None, None, None]
            )
        )

        var = coef.posterior_variance[t][:, None, None, None]

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
