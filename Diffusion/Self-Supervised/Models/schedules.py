import torch
import math


def make_beta_schedule(T: int, schedule: str = "linear", beta_start=1e-4, beta_end=2e-2):
    """
    Creates a beta schedule for diffusion processes.

    Args:
        T (int): Number of timesteps.
        schedule (str): Type of schedule ("linear" or "cosine").
        beta_start (float): Starting beta value for linear schedule.
        beta_end (float): Ending beta value for linear schedule.

    Returns:
        torch.Tensor: A tensor of betas with shape [T].
    """
    if schedule == "linear":
        return torch.linspace(beta_start, beta_end, T)

    elif schedule == "cosine":
        # Nichol & Dhariwal cosine schedule in alpha_bar space
        t = torch.arange(T + 1, dtype=torch.float64) / T
        s = 0.008
        f = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        alpha_bar = f / f[0]
        betas = 1 - (alpha_bar[1:] / alpha_bar[:-1])
        return betas.to(dtype=torch.float32).clamp(1e-8, 0.999)

    else:
        raise ValueError(f"Unknown schedule: {schedule}")


class DiffusionCoeffs:
    """
    Precomputes useful diffusion coefficients given a beta schedule.
    """

    def __init__(self, betas: torch.Tensor):
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)
        self.sqrt_alphas = torch.sqrt(self.alphas)
        self.one_over_sqrt_alphas = 1.0 / self.sqrt_alphas

        # Posterior variance (DDPM)
        alphas_cumprod_prev = torch.cat(
            [torch.tensor([1.0], dtype=betas.dtype), self.alpha_bars[:-1]], dim=0
        )
        self.posterior_variance = (
            betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alpha_bars)
        )
        self.posterior_log_variance_clipped = torch.log(
            self.posterior_variance.clamp(min=1e-20)
        )

        self.posterior_mean_coef1 = (
            torch.sqrt(alphas_cumprod_prev) * betas / (1.0 - self.alpha_bars)
        )
        self.posterior_mean_coef2 = (
            torch.sqrt(self.alphas)
            * (1.0 - alphas_cumprod_prev)
            / (1.0 - self.alpha_bars)
        )
