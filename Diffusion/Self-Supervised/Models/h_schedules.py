# models/h_schedules.py
import math
import torch


def make_beta_schedule(T=1000, mode='cosine', beta_start=1e-4, beta_end=2e-2):
    """
    Create a beta schedule for diffusion processes.

    Args:
        T (int): number of timesteps.
        mode (str): 'linear' or 'cosine'.
        beta_start (float): start value for linear schedule.
        beta_end (float): end value for linear schedule.

    Returns:
        betas: Tensor of shape [T]
        alphas: 1 - betas
        alpha_bars: cumulative product of alphas
    """
    if mode == 'linear':
        betas = torch.linspace(beta_start, beta_end, T)
    elif mode == 'cosine':
        t = torch.linspace(0, 1, T + 1, dtype=torch.float64)
        s = 0.008
        f = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        a_bar = (f / f[0]).clamp(min=1e-5)
        betas = (1 - a_bar[1:] / a_bar[:-1]).to(torch.float32).clamp(1e-8, 0.999)
    else:
        raise ValueError(f"Unknown schedule mode: {mode}")

    alphas = 1 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars


@torch.no_grad()
def ddim_step_eps(x, t, t_prev, eps_hat, alpha_bars):
    """
    Perform a single DDIM step using predicted epsilon.

    Args:
        x (Tensor): x_t at current timestep.
        t (int): current timestep index.
        t_prev (int): previous timestep index.
        eps_hat (Tensor): predicted noise.
        alpha_bars (Tensor): cumulative product of alphas.

    Returns:
        x_prev (Tensor): reconstructed x at previous timestep.
    """
    ab_t = alpha_bars[t]
    ab_s = alpha_bars[t_prev]
    
    # Estimate x_0 from predicted epsilon
    x0 = (x - (1 - ab_t).sqrt() * eps_hat) / ab_t.sqrt()
    
    # Compute the direction term for DDIM
    dir_term = (1 - ab_s).sqrt() * eps_hat
    
    # Compute x_{t_prev}
    x_prev = ab_s.sqrt() * x0 + dir_term
    return x_prev
