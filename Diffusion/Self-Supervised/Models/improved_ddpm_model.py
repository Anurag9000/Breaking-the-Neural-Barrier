# models/improved_ddpm_model.py
import torch
import torch.nn as nn
from models.common_unet import DenoiserUNet
from models.schedules import make_beta_schedule, DiffusionCoeffs


class ImprovedDDPM(nn.Module):
    """
    Improved DDPM implementation with SNR-weighted loss (optional)
    and cosine beta schedule (default).

    Reference:
        "Improved Denoising Diffusion Probabilistic Models" – Nichol & Dhariwal (2021)
    """

    def __init__(
        self,
        img_size=32,
        channels=3,
        T=1000,
        schedule="cosine",
        base=64,
        loss_weight="uniform",
    ):
        super().__init__()
        self.T = T
        self.channels = channels
        self.loss_weight = loss_weight  # 'uniform' or 'snr'

        # Model and diffusion coefficients
        self.net = DenoiserUNet(base=base, in_ch=channels, out_ch=channels)
        betas = make_beta_schedule(T, schedule)
        self.register_buffer("betas", betas)
        self.coeff = DiffusionCoeffs(betas)

    # ----------------------------------------------------------
    # Forward diffusion: q(x_t | x_0)
    # ----------------------------------------------------------
    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return (
            self.coeff.sqrt_alpha_bars[t][:, None, None, None] * x0
            + self.coeff.sqrt_one_minus_alpha_bars[t][:, None, None, None] * noise
        )

    # ----------------------------------------------------------
    # Loss weighting scheme
    # ----------------------------------------------------------
    def _loss_weight(self, t: torch.Tensor) -> torch.Tensor:
        if self.loss_weight == "uniform":
            return torch.ones_like(t, dtype=torch.float32)

        # SNR weighting: w = snr / (snr + 1)
        alpha_bar = self.coeff.alpha_bars[t]
        snr = alpha_bar / (1 - alpha_bar + 1e-8)
        return (snr / (snr + 1)).float()

    # ----------------------------------------------------------
    # Training loss: predict ε (noise)
    # ----------------------------------------------------------
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        device = x0.device

        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)
        x_t = self.q_sample(x0, t, noise)
        eps_hat = self.net(x_t, t)

        # Per-sample MSE loss
        base_loss = (eps_hat - noise).pow(2).mean(dim=(1, 2, 3))

        # Weight by uniform or SNR-based factor
        w = self._loss_weight(t)
        return (base_loss * w).mean()

    # ----------------------------------------------------------
    # Reverse diffusion: p(x_{t-1} | x_t)
    # ----------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        eps = self.net(x_t, t)
        c = self.coeff

        mean = (
            c.posterior_mean_coef1[t][:, None, None, None] * x_t
            + c.posterior_mean_coef2[t][:, None, None, None]
            * (
                (x_t - c.sqrt_one_minus_alpha_bars[t][:, None, None, None] * eps)
                / c.sqrt_alpha_bars[t][:, None, None, None]
            )
        )

        var = c.posterior_variance[t][:, None, None, None]

        # No noise at t = 0
        if (t == 0).all():
            return mean

        return mean + torch.randn_like(x_t) * torch.sqrt(var)

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
