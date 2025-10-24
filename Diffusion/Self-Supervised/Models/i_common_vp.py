import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Beta schedule (linear or cosine)
# ---------------------------
def beta_schedule(T=1000, mode='cosine', beta_start=1e-4, beta_end=2e-2):
    if mode == 'linear':
        betas = torch.linspace(beta_start, beta_end, T)
    elif mode == 'cosine':
        t = torch.linspace(0, 1, T + 1, dtype=torch.float64)
        s = 0.008
        f = torch.cos(((t + s) / (1 + s)) * math.pi / 2) ** 2
        a_bar = (f / f[0]).clamp(min=1e-5)
        betas = (1 - a_bar[1:] / a_bar[:-1]).to(torch.float32).clamp(1e-8, 0.999)
    else:
        raise ValueError(f'Unknown beta schedule mode: {mode}')

    alphas = 1 - betas
    a_bar = torch.cumprod(alphas, dim=0)
    return betas, alphas, a_bar

# ---------------------------
# Sinusoidal time embedding
# ---------------------------
def time_embed(t, dim=256):
    device = t.device
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1)))
    ang = t.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb

# ---------------------------
# FiLM layer
# ---------------------------
class FiLM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_dim * 2)
        )

    def forward(self, c):
        a, b = self.mlp(c).chunk(2, dim=1)
        return a, b

# ---------------------------
# Residual block with FiLM
# ---------------------------
class ResBlk(nn.Module):
    def __init__(self, ci, co, cd):
        super().__init__()
        self.c1 = nn.Conv2d(ci, co, 3, padding=1)
        self.g1 = nn.GroupNorm(8, co)
        self.c2 = nn.Conv2d(co, co, 3, padding=1)
        self.g2 = nn.GroupNorm(8, co)
        self.f = FiLM(cd, co)
        self.skip = nn.Identity() if ci == co else nn.Conv2d(ci, co, 1)

    def forward(self, x, c):
        a, b = self.f(c)
        h = self.c1(x)
        h = self.g1(h)
        h = F.silu(h)
        h = self.c2(h)
        h = self.g2(h)
        h = h * (1 + a[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(h)
        return h + self.skip(x)

# ---------------------------
# Downsampling block
# ---------------------------
class Down(nn.Module):
    def __init__(self, ci, co, cd):
        super().__init__()
        self.b = ResBlk(ci, co, cd)
        self.d = nn.Conv2d(co, co, 3, stride=2, padding=1)

    def forward(self, x, c):
        x = self.b(x, c)
        return self.d(x)

# ---------------------------
# Upsampling block
# ---------------------------
class Up(nn.Module):
    def __init__(self, ci, co, cd):
        super().__init__()
        self.b = ResBlk(ci, co, cd)
        self.u = nn.ConvTranspose2d(co, co, 4, stride=2, padding=1)

    def forward(self, x, c):
        x = self.b(x, c)
        x = self.u(x)
        return x
