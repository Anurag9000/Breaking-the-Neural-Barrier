from dataclasses import dataclass
import torch, math
import torch.nn as nn
import torch.nn.functional as F


# -----------------------------
# Configuration
# -----------------------------
@dataclass
class DiffusionConfig:
    T: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2
    objective: str = "eps"  # "eps" or "v"
    p_uncond: float = 0.0   # dropout conditioning (0 = always conditioned)


# -----------------------------
# Sinusoidal Time Embedding
# -----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.proj = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        # t: (b,) in [0, T-1]
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device) * (-math.log(10000.0) / (half - 1))
        )
        args = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))
        return self.proj(emb)


# -----------------------------
# Beta Schedule
# -----------------------------
class BetaSchedule:
    def __init__(self, T, beta_start, beta_end):
        betas = torch.linspace(beta_start, beta_end, T)
        alphas = 1.0 - betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.register(T, betas, alphas, alphas_cum)

    def register(self, T, betas, alphas, alphas_cum):
        self.T = T
        self.betas = betas
        self.alphas = alphas
        self.alphas_cum = alphas_cum
        self.sqrt_alphas_cum = torch.sqrt(alphas_cum)
        self.sqrt_one_minus = torch.sqrt(1.0 - alphas_cum)
        self.inv_sqrt_alphas = torch.sqrt(1.0 / alphas)


# -----------------------------
# Diffusion Utilities
# -----------------------------
def q_sample(x0, t, sched, noise=None):
    if noise is None:
        noise = torch.randn_like(x0)
    a = sched.sqrt_alphas_cum[t].view(-1, 1, 1, 1)
    b = sched.sqrt_one_minus[t].view(-1, 1, 1, 1)
    return a * x0 + b * noise, noise


def to_v(x0, eps, t, sched):
    a = sched.sqrt_alphas_cum[t].view(-1, 1, 1, 1)
    b = sched.sqrt_one_minus[t].view(-1, 1, 1, 1)
    return a * eps - b * x0


def from_v(v, t, sched, eps=None, x0=None):
    # recover eps if x0 given, else recover x0 if eps given
    a = sched.sqrt_alphas_cum[t].view(-1, 1, 1, 1)
    b = sched.sqrt_one_minus[t].view(-1, 1, 1, 1)
    if x0 is not None:
        return (v + b * x0) / a
    if eps is not None:
        return (v - a * eps) / b
    raise ValueError("Need eps or x0 with v-prediction")


# -----------------------------
# Diffusion Loss
# -----------------------------
class DiffusionLoss(nn.Module):
    def __init__(self, cfg: DiffusionConfig, sched: BetaSchedule):
        super().__init__()
        self.cfg = cfg
        self.sched = sched

    def forward(self, model, x0, cond, times):
        x_t, eps = q_sample(x0, times, self.sched)
        if self.cfg.objective == "eps":
            pred = model(x_t, times, cond)
            return F.mse_loss(pred, eps)
        else:  # v-objective
            v = to_v(x0, eps, times, self.sched)
            pred = model(x_t, times, cond)
            return F.mse_loss(pred, v)
