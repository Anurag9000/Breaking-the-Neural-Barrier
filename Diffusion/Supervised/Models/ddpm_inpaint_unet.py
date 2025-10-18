import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# UNet with mask+context conditioning concatenated at input
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        device = t.device
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class MLPTime(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.SiLU(), nn.Linear(out_dim, out_dim))

    def forward(self, t_emb: torch.Tensor) -> torch.Tensor:
        return self.net(t_emb)

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, t_dim: int, n_groups: int = 8):
        super().__init__()
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

class InpaintDDPMUNet(nn.Module):
    """Inpainting DDPM UNet (epsilon prediction).
    Inputs: x_t (3ch) and conditioning [context_img(3ch), mask(1ch)] concatenated => total 7 channels.
    """
    def __init__(self, base_ch: int = 64, ch_mults=(1,2,2,4), time_dim: int = 256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = MLPTime(time_dim, time_dim)
        self.in_conv = nn.Conv2d(3 + 3 + 1, base_ch, 3, padding=1)
        self.downs = nn.ModuleList()
        ch = base_ch
        self.skips = []
        for m in ch_mults:
            out_ch = base_ch * m
            self.downs.append(Down(ch, out_ch, time_dim, use_attn=False))
            ch = out_ch
        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, time_dim)
        self.ups = nn.ModuleList()
        for m in reversed(ch_mults):
            out_ch = base_ch * m
            self.ups.append(Up(ch, out_ch, time_dim, use_attn=False))
            ch = out_ch
        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, 3, 3, padding=1)

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, context_img: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        t_feat = self.time_mlp(self.time_emb(t))
        x = torch.cat([x_t, context_img, mask], dim=1)
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
        return eps

    @torch.no_grad()
    def sample(self, context_img: torch.Tensor, mask: torch.Tensor, betas: torch.Tensor, device: Optional[torch.device] = None, eta: float = 0.0):
        device = device or next(self.parameters()).device
        b, _, h, w = context_img.shape
        x = torch.randn((b, 3, h, w), device=device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        sqrt_ac = torch.sqrt(ac)
        sqrt_om = torch.sqrt(1 - ac)
        # during sampling, keep known pixels from context_img fixed via mask
        for i in reversed(range(len(betas))):
            t = torch.full((b,), (i + 0.5) / len(betas), device=device)
            eps = self.forward(x, t, context_img, mask)
            x0 = (x - sqrt_om[i] * eps) / (sqrt_ac[i] + 1e-8)
            if i == 0:
                x = x0
            else:
                a_prev = ac[i - 1]
                sigma = eta * math.sqrt((1 - a_prev) / (1 - ac[i]) * (1 - ac[i] / a_prev))
                noise = torch.randn_like(x) if sigma > 0 else 0.0
                x = torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev - sigma ** 2) * eps + sigma * noise
            # enforce known pixels
            x = x * mask + context_img * (1 - mask)
        return x.clamp(-1, 1)
