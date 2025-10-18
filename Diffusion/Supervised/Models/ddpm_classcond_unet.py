import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Positional/time embeddings
# -----------------------------
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t in [0, 1]; map to radians range
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(
            torch.linspace(math.log(1.0), math.log(10000.0), half, device=device)
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class MLPTime(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim)
        )

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)

# -----------------------------
# UNet building blocks
# -----------------------------
class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, n_groups: int = 8):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.norm1 = nn.GroupNorm(n_groups, in_ch)
        self.act = nn.SiLU()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)

        self.time = nn.Linear(t_dim, out_ch)

        self.norm2 = nn.GroupNorm(n_groups, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)

        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_feat: torch.Tensor) -> torch.Tensor:
        h = self.conv1(self.act(self.norm1(x)))
        h = h + self.time(t_feat)[:, :, None, None]
        h = self.conv2(self.act(self.norm2(h)))
        return h + self.skip(x)

class AttnBlock(nn.Module):
    def __init__(self, ch: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.GroupNorm(8, ch)
        self.q = nn.Conv2d(ch, ch, 1)
        self.k = nn.Conv2d(ch, ch, 1)
        self.v = nn.Conv2d(ch, ch, 1)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.num_heads = num_heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        h_groups = self.num_heads
        x_norm = self.norm(x)
        q = self.q(x_norm).view(b, h_groups, c // h_groups, h * w)
        k = self.k(x_norm).view(b, h_groups, c // h_groups, h * w)
        v = self.v(x_norm).view(b, h_groups, c // h_groups, h * w)
        attn = torch.einsum('bhcn,bhcm->bhnm', q, k) / math.sqrt(c // h_groups)
        attn = attn.softmax(dim=-1)
        out = torch.einsum('bhnm,bhcm->bhcn', attn, v)
        out = out.reshape(b, c, h, w)
        return x + self.proj(out)

class Down(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, use_attn: bool):
        super().__init__()
        self.block1 = ResBlock(in_ch, out_ch, t_dim)
        self.block2 = ResBlock(out_ch, out_ch, t_dim)
        self.attn = AttnBlock(out_ch) if use_attn else nn.Identity()
        self.down = nn.Conv2d(out_ch, out_ch, 3, stride=2, padding=1)

    def forward(self, x, t):
        x = self.block1(x, t)
        x = self.block2(x, t)
        x = self.attn(x)
        skip = x
        x = self.down(x)
        return x, skip

class Up(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, use_attn: bool):
        super().__init__()
        self.block1 = ResBlock(in_ch + out_ch, out_ch, t_dim)
        self.block2 = ResBlock(out_ch, out_ch, t_dim)
        self.attn = AttnBlock(out_ch) if use_attn else nn.Identity()
        self.up = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)

    def forward(self, x, skip, t):
        x = torch.cat([x, skip], dim=1)
        x = self.block1(x, t)
        x = self.block2(x, t)
        x = self.attn(x)
        x = self.up(x)
        return x

# -----------------------------
# Class-conditional UNet for DDPM (epsilon prediction)
# -----------------------------
class DDPMClassCondUNet(nn.Module):
    def __init__(self, in_ch: int = 3, base_ch: int = 64, ch_mults=(1, 2, 2, 4),
                 num_classes: int = 10, time_dim: int = 256, attn_res=(16,)):
        super().__init__()
        self.in_ch = in_ch
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = MLPTime(time_dim, time_dim)
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)  # +1 for null label (CFG)

        # in
        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # downs
        self.downs = nn.ModuleList()
        ch = base_ch
        self.skips_ch = []
        res_map = []
        for i, m in enumerate(ch_mults):
            out_ch = base_ch * m
            use_attn = False
            res_map.append(use_attn)
            self.downs.append(Down(ch, out_ch, time_dim, use_attn))
            self.skips_ch.append(out_ch)
            ch = out_ch

        # mid
        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, time_dim)

        # ups (mirror)
        self.ups = nn.ModuleList()
        for m, skip_ch in zip(reversed(ch_mults), reversed(self.skips_ch)):
            out_ch = base_ch * m
            self.ups.append(Up(ch, out_ch, time_dim, False))
            ch = out_ch

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # t: float in [0,1], y: int in [0..num_classes] (num_classes => null for CFG)
        t_feat = self.time_mlp(self.time_emb(t))
        # Combine time + label embedding into single conditioning vector (additive)
        y_emb = self.label_emb(y)
        t_feat = t_feat + y_emb

        x = self.in_conv(x)
        skips = []
        for d in self.downs:
            x, s = d(x, t_feat)
            skips.append(s)
        x = self.mid1(x, t_feat)
        x = self.mid_attn(x)
        x = self.mid2(x, t_feat)
        for u in self.ups:
            s = skips.pop()
            x = u(x, s, t_feat)
        x = F.silu(self.out_norm(x))
        eps = self.out_conv(x)
        return eps  # predict epsilon

    @torch.no_grad()
    def sample(self, shape, num_steps: int, betas: torch.Tensor, y: torch.Tensor, cfg_scale: float = 0.0,
               device: Optional[torch.device] = None, eta: float = 0.0):
        """DDPM/DDIM sampling (eta=0 -> DDIM deterministic). y is class labels; cfg via null label index.
        """
        device = device or next(self.parameters()).device
        b = shape[0]
        x = torch.randn(shape, device=device)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)

        null_label = torch.full_like(y, self.label_emb.num_embeddings - 1)

        for i in reversed(range(num_steps)):
            t = torch.full((b,), (i + 0.5) / num_steps, device=device)
            # CFG: run conditional and unconditional in the same model (single-model compliant)
            eps_cond = self.forward(x, t, y)
            if cfg_scale > 0:
                eps_null = self.forward(x, t, null_label)
                eps = eps_null + cfg_scale * (eps_cond - eps_null)
            else:
                eps = eps_cond

            alpha_t = alphas_cumprod[i]
            sqrt_alpha_t = sqrt_alphas_cumprod[i]
            sqrt_one_minus_alpha_t = sqrt_one_minus_alphas_cumprod[i]

            x0_pred = (x - sqrt_one_minus_alpha_t * eps) / (sqrt_alpha_t + 1e-8)

            if i == 0:
                x = x0_pred
            else:
                # DDIM update
                t_prev = i - 1
                alpha_prev = alphas_cumprod[t_prev]
                sigma_t = eta * math.sqrt(
                    (1 - alpha_prev) / (1 - alpha_t) * (1 - alpha_t / alpha_prev)
                )
                dir_xt = torch.sqrt(alpha_prev) * x0_pred
                noise = torch.randn_like(x) if sigma_t > 0 else 0.0
                x = dir_xt + torch.sqrt(1 - alpha_prev - sigma_t ** 2) * eps + sigma_t * noise
        return x.clamp(-1, 1)
