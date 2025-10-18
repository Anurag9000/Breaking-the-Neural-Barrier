import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Positional / time embeddings
# -----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor):
        # t: (B,) in [0, T-1]
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device, dtype=t.dtype) * -(math.log(10000.0) / (half - 1))
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0,1))
        return emb

# -----------------------------
# UNet backbone (minimal, single-model)
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim, *, dropout=0.0):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.time_mlp = nn.Sequential(
            nn.SiLU(), nn.Linear(time_dim, out_ch)
        )
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.act(h)
        # add time
        temb = self.time_mlp(t_emb)[:, :, None, None]
        h = h + temb
        h = self.conv2(h)
        h = self.bn2(h)
        h = self.dropout(h)
        out = self.act(h + self.skip(x))
        return out

class Down(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block1 = ResBlock(in_ch, out_ch, time_dim)
        self.block2 = ResBlock(out_ch, out_ch, time_dim)
        self.pool = nn.AvgPool2d(2)

    def forward(self, x, t):
        x = self.block1(x, t)
        x = self.block2(x, t)
        down = self.pool(x)
        return x, down

class Up(nn.Module):
    def __init__(self, in_ch, out_ch, time_dim):
        super().__init__()
        self.block1 = ResBlock(in_ch, out_ch, time_dim)
        self.block2 = ResBlock(out_ch, out_ch, time_dim)

    def forward(self, x, skip, t):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t)
        x = self.block2(x, t)
        return x

class UNet(nn.Module):
    def __init__(self, in_ch=3, base=64, ch_mult=(1,2,4), time_dim=256, out_ch=3):
        super().__init__()
        self.time_emb = nn.Sequential(
            SinusoidalTimeEmbedding(time_dim),
            nn.Linear(time_dim, time_dim*4), nn.SiLU(),
            nn.Linear(time_dim*4, time_dim)
        )
        c1, c2, c3 = [base*m for m in ch_mult]
        self.in_conv = nn.Conv2d(in_ch, c1, 3, padding=1)

        # Down path
        self.down1 = Down(c1, c1, time_dim)
        self.down2 = Down(c1, c2, time_dim)
        self.mid1  = ResBlock(c2, c3, time_dim)
        self.mid2  = ResBlock(c3, c3, time_dim)

        # Up path
        self.up1 = Up(c3 + c2, c2, time_dim)
        self.up2 = Up(c2 + c1, c1, time_dim)
        self.out_norm = nn.BatchNorm2d(c1)
        self.out_act = nn.SiLU()
        self.out_conv = nn.Conv2d(c1, out_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.time_emb(t)
        x0 = self.in_conv(x)
        s1, d1 = self.down1(x0, t_emb)
        s2, d2 = self.down2(d1, t_emb)
        m  = self.mid1(d2, t_emb)
        m  = self.mid2(m, t_emb)
        u1 = self.up1(m, s2, t_emb)
        u2 = self.up2(u1, s1, t_emb)
        h  = self.out_norm(u2)
        h  = self.out_act(h)
        out = self.out_conv(h)
        return out

# -----------------------------
# Diffusion (DDPM, epsilon prediction)
# -----------------------------
@dataclass
class DiffusionConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 2e-2

class DDPM(nn.Module):
    def __init__(self, model: nn.Module, cfg: DiffusionConfig):
        super().__init__()
        self.model = model
        self.cfg = cfg
        self.register_buffer('betas', torch.linspace(cfg.beta_start, cfg.beta_end, cfg.timesteps))
        alphas = 1.0 - self.betas
        alphas_cum = torch.cumprod(alphas, dim=0)
        self.register_buffer('alphas_cumprod', alphas_cum)
        self.register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cum))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', torch.sqrt(1.0 - alphas_cum))

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor]=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_ac = self.sqrt_alphas_cumprod[t][:, None, None, None]
        sqrt_om = self.sqrt_one_minus_alphas_cumprod[t][:, None, None, None]
        return sqrt_ac * x0 + sqrt_om * noise, noise

    def p_losses(self, x0: torch.Tensor, t: torch.Tensor):
        x_t, noise = self.q_sample(x0, t)
        eps_pred = self.model(x_t, t.float())
        return F.mse_loss(eps_pred, noise)

    @torch.no_grad()
    def p_sample(self, x: torch.Tensor, t: int):
        betat = self.betas[t]
        ac_t = self.alphas_cumprod[t]
        ac_tm1 = self.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=x.device)
        coef1 = 1 / torch.sqrt(1 - betat)
        eps = self.model(x, torch.full((x.size(0),), float(t), device=x.device))
        mean = (1/torch.sqrt(ac_t)) * (x - (betat/torch.sqrt(1 - ac_t)) * eps)
        if t > 0:
            noise = torch.randn_like(x)
            sigma = torch.sqrt((1 - ac_tm1) / (1 - ac_t) * betat)
            x_prev = mean + sigma * noise
        else:
            x_prev = mean
        return x_prev

    @torch.no_grad()
    def sample(self, shape, device):
        x = torch.randn(shape, device=device)
        T = self.cfg.timesteps
        for t in reversed(range(T)):
            x = self.p_sample(x, t)
        return x.clamp(-1, 1)

# Utility to count params (parity with your STL helper style)

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
