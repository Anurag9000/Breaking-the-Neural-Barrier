import math
from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------- Time Embedding ----------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    def forward(self, t: torch.Tensor):
        half = self.dim // 2
        freqs = torch.exp(torch.arange(half, device=t.device) * -(math.log(10000.0)/(half-1)))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2:
            emb = F.pad(emb, (0,1))
        return emb

# ---------- UNet backbone (score network) ----------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, tdim):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.emb = nn.Sequential(nn.SiLU(), nn.Linear(tdim, out_ch))
        self.act = nn.SiLU()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()
    def forward(self, x, temb):
        h = self.act(self.bn1(self.conv1(x)))
        h = h + self.emb(temb)[:, :, None, None]
        h = self.bn2(self.conv2(h))
        return self.act(h + self.skip(x))

class Down(nn.Module):
    def __init__(self, c_in, c_out, tdim):
        super().__init__()
        self.b1 = ResBlock(c_in, c_out, tdim)
        self.b2 = ResBlock(c_out, c_out, tdim)
        self.pool = nn.AvgPool2d(2)
    def forward(self, x, t):
        x = self.b1(x,t); x = self.b2(x,t)
        d = self.pool(x)
        return x, d

class Up(nn.Module):
    def __init__(self, c_in, c_out, tdim):
        super().__init__()
        self.b1 = ResBlock(c_in, c_out, tdim)
        self.b2 = ResBlock(c_out, c_out, tdim)
    def forward(self, x, skip, t):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.b1(x,t); x = self.b2(x,t)
        return x

class ScoreUNet(nn.Module):
    def __init__(self, in_ch=3, base=64, ch_mult=(1,2,4), tdim=256, out_ch=3):
        super().__init__()
        self.tproj = nn.Sequential(
            SinusoidalTimeEmbedding(tdim), nn.Linear(tdim, tdim*4), nn.SiLU(), nn.Linear(tdim*4, tdim)
        )
        c1,c2,c3 = [base*m for m in ch_mult]
        self.in_conv = nn.Conv2d(in_ch, c1, 3, padding=1)
        self.d1 = Down(c1, c1, tdim)
        self.d2 = Down(c1, c2, tdim)
        self.mid1 = ResBlock(c2, c3, tdim)
        self.mid2 = ResBlock(c3, c3, tdim)
        self.u1 = Up(c3+c2, c2, tdim)
        self.u2 = Up(c2+c1, c1, tdim)
        self.out_bn = nn.BatchNorm2d(c1)
        self.out = nn.Conv2d(c1, out_ch, 3, padding=1)
        self.act = nn.SiLU()
    def forward(self, x, t):
        t = self.tproj(t)
        x0 = self.in_conv(x)
        s1,d1 = self.d1(x0,t)
        s2,d2 = self.d2(d1,t)
        m = self.mid1(d2,t); m = self.mid2(m,t)
        u1 = self.u1(m,s2,t)
        u2 = self.u2(u1,s1,t)
        h = self.act(self.out_bn(u2))
        return self.out(h)  # score field (same channels as input)

# ---------- SDE utilities ----------
@dataclass
class SDEConfig:
    sigma_min: float = 0.01
    sigma_max: float = 50.0
    continuous: bool = True

class VPSDE:
    def __init__(self, T=1.0, beta_min=0.1, beta_max=20.0):
        self.T=T; self.beta_min=beta_min; self.beta_max=beta_max
    def marginal_prob(self, x0: torch.Tensor, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # mean = x0 * exp(-0.5 * ∫ beta dt), std^2 = 1 - exp(-∫ beta dt)
        log_mean_coeff = -0.25*(self.beta_max - self.beta_min)*t**2 - 0.5*self.beta_min*t
        mean = torch.exp(log_mean_coeff)[:,None,None,None] * x0
        std = torch.sqrt(1.0 - torch.exp(2*log_mean_coeff))[:,None,None,None]
        return mean, std

# Training target: score(x_t,t) = ∇_x log p(x_t|x0) = -(x_t - mean)/std^2
class ScoreSDE(nn.Module):
    def __init__(self, score_net: nn.Module, sde: VPSDE):
        super().__init__()
        self.net = score_net
        self.sde = sde
    def loss(self, x0: torch.Tensor):
        B = x0.size(0)
        t = torch.rand(B, device=x0.device) * self.sde.T
        mean, std = self.sde.marginal_prob(x0, t)
        noise = torch.randn_like(x0)
        x_t = mean + std * noise
        target = -(x_t - mean) / (std**2 + 1e-8)
        # time embedding expects shape (B,), pass raw t
        pred = self.net(x_t, t)
        return F.mse_loss(pred, target)
    @torch.no_grad()
    def pc_sample(self, shape, device, steps=50):
        # Predictor-Corrector sampler (simplified, single model)
        x = torch.randn(shape, device=device)
        for i in reversed(range(steps)):
            t = torch.full((shape[0],), (i+1)/steps, device=device)
            mean, std = self.sde.marginal_prob(torch.zeros_like(x), t)
            score = self.net(x, t)
            # Euler-Maruyama predictor (heuristic step)
            dt = 1.0/steps
            x = x + (std**2) * score * dt
            # Corrector (Langevin)
            noise = torch.randn_like(x)
            x = x + 0.01 * score + math.sqrt(0.02) * noise
        return x.clamp(-1,1)

def count_parameters(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
