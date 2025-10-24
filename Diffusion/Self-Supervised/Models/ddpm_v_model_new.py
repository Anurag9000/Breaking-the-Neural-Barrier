import torch
import torch.nn as nn
from models.common_unet import DenoiserUNet
from models.schedules import make_beta_schedule, DiffusionCoeffs


class DDPMV(nn.Module):
    """
    DDPM variant that predicts the velocity vector `v`, as introduced in
    Imagen and Stable Diffusion. This parameterization improves training stability.
    """

    def __init__(self, img_size=32, channels=3, T=1000, schedule="linear", base=64):
        super().__init__()
        self.T = T
        self.channels = channels

        # Network and diffusion coefficients
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
    # Training: predict v (velocity) instead of ε or x₀
    # ----------------------------------------------------------
    def loss(self, x0: torch.Tensor) -> torch.Tensor:
        B = x0.size(0)
        device = x0.device

        # Random time step and noise
        t = torch.randint(0, self.T, (B,), device=device, dtype=torch.long)
        noise = torch.randn_like(x0)

        # Forward sample
        x_t = self.q_sample(x0, t, noise)

        # Define alpha, sigma in log-SNR style
        alpha_bar = self.coeff.alpha_bars[t]
        alpha = torch.sqrt(alpha_bar)
        sigma = torch.sqrt(1 - alpha_bar)

        # v = α * ε - σ * x₀
        v_target = alpha[:, None, None, None] * noise - sigma[:, None, None, None] * x0
        v_hat = self.net(x_t, t)

        return nn.functional.mse_loss(v_hat, v_target)

    # ----------------------------------------------------------
    # Helper: recover x₀ and ε from v
    # ----------------------------------------------------------
    @torch.no_grad()
    def _x0_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor):
        alpha_bar = self.coeff.alpha_bars[t]
        alpha = torch.sqrt(alpha_bar)[:, None, None, None]
        sigma = torch.sqrt(1 - alpha_bar)[:, None, None, None]

        # From definitions: v = α * ε - σ * x₀
        # Solve for x₀ and ε
        x0 = (x_t + sigma * v) / alpha
        eps = (v + sigma * x0) / alpha

        return x0, eps

    # ----------------------------------------------------------
    # Reverse diffusion: p(x_{t-1} | x_t)
    # ----------------------------------------------------------
    @torch.no_grad()
    def p_sample(self, x_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        v = self.net(x_t, t)
        x0, eps = self._x0_eps_from_v(x_t, t, v)
        c = self.coeff

        mean = (
            c.posterior_mean_coef1[t][:, None, None, None] * x_t
            + c.posterior_mean_coef2[t][:, None, None, None] * x0
        )
        var = c.posterior_variance[t][:, None, None, None]

        # If t = 0, don't add noise
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
