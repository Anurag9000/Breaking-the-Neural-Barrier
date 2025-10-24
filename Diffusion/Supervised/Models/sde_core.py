from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F


# ====== VP-SDE (Variance Preserving) ======
def vp_marginal_params(t, cfg):
    at = torch.exp(-0.5 * beta_t(t, cfg))
    sigma2 = 1.0 - at * at
    return at, sigma2


# ====== VE-SDE (Variance Exploding) ======
# d x = σ(t) dW, with σ(t) geometric from σ_min to σ_max
def ve_sigma(t, cfg):
    return cfg.sigma_min * (cfg.sigma_max / cfg.sigma_min) ** t


# ====== Perturb & Targets ======
def perturb_vp(x0, t, cfg):
    a, s2 = vp_marginal_params(t, cfg)
    n = torch.randn_like(x0)
    x_t = a.view(-1, 1, 1, 1) * x0 + (s2.clamp(1e-12).sqrt()).view(-1, 1, 1, 1) * n
    target_score = -(n / (s2.view(-1, 1, 1, 1).sqrt() + 1e-12))  # ∇_{x_t} log p(x_t|x0)
    return x_t, target_score


def perturb_ve(x0, t, cfg):
    sig = ve_sigma(t, cfg)
    n = torch.randn_like(x0)
    x_t = x0 + sig.view(-1, 1, 1, 1) * n
    target_score = -n / (sig.view(-1, 1, 1, 1) + 1e-12)
    return x_t, target_score


# ====== Losses ======
class ScoreLoss(nn.Module):
    """Simple MSE between predicted score and target score with per-t weighting."""

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def forward(self, model, x0, cond, t):
        if self.cfg.type == "vp":
            x_t, score_tgt = perturb_vp(x0, t, self.cfg)
        else:
            x_t, score_tgt = perturb_ve(x0, t, self.cfg)

        pred = model(x_t, t, cond)
        return F.mse_loss(pred, score_tgt)


# ====== Samplers ======
@torch.no_grad()
def euler_maruyama_sampler(model, x_T, cond, cfg, steps=100):
    x = x_T
    ts = torch.linspace(cfg.T, 0.0, steps, device=x.device)
    for i in range(len(ts) - 1):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        tt = torch.full((x.size(0),), t, device=x.device)
        score = model(x, tt, cond)

        if cfg.type == "vp":
            b = beta_t(t, cfg)
            drift = -0.5 * b * x + b.sqrt() ** 2 * score
            diffusion = b.sqrt()
        else:  # VE
            sig = ve_sigma(t, cfg)
            drift = sig ** 2 * score
            diffusion = sig

        x = x + drift * dt + diffusion * (dt.abs().sqrt()) * torch.randn_like(x)
    return x


@torch.no_grad()
def probability_flow_ode(model, x_T, cond, cfg, steps=100):
    """Deterministic sampler using the probability flow ODE."""
    x = x_T
    ts = torch.linspace(cfg.T, 0.0, steps, device=x.device)
    for i in range(len(ts) - 1):
        t = ts[i]
        dt = ts[i + 1] - ts[i]
        tt = torch.full((x.size(0),), t, device=x.device)
        score = model(x, tt, cond)

        if cfg.type == "vp":
            b = beta_t(t, cfg)
            drift = -0.5 * b * x + b * score
        else:
            sig = ve_sigma(t, cfg)
            drift = sig ** 2 * score

        x = x + drift * dt
    return x
