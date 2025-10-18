import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal DiT-like transformer denoiser for 32x32 images (CIFAR-10)
# Single-model, class-conditional via label token embedding and classifier-free guidance support.

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, patch=4, dim=256):
        super().__init__()
        self.patch = patch
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=patch)

    def forward(self, x):
        # B,C,H,W -> B, N, D
        x = self.proj(x)
        b, d, h, w = x.shape
        x = x.flatten(2).transpose(1, 2)
        return x, (h, w)

class PositionalEncoding(nn.Module):
    def __init__(self, dim: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)  # [max_len, dim]

    def forward(self, x):
        n = x.size(1)
        return x + self.pe[:n, :]

class DiTBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 8, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(), nn.Linear(int(dim * mlp_ratio), dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + self.mlp(self.norm2(x))
        return x

class TimeEmbedding(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, t_vec):
        return self.lin(t_vec)

class SinTime(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device))
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        if self.dim % 2 == 1:
            emb = F.pad(emb, (0, 1))
        return emb

class DiTDenoiser(nn.Module):
    def __init__(self, in_ch=3, num_classes=10, patch=4, dim=256, depth=8, n_heads=8):
        super().__init__()
        self.patch = patch
        self.num_classes = num_classes
        self.patch_embed = PatchEmbed(in_ch, patch, dim)
        self.pos = PositionalEncoding(dim)
        self.time_sin = SinTime(dim)
        self.time_mlp = TimeEmbedding(dim)
        self.label_emb = nn.Embedding(num_classes + 1, dim)  # +1 for null

        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.time_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.label_token = nn.Parameter(torch.zeros(1, 1, dim))

        self.blocks = nn.ModuleList([DiTBlock(dim, n_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        # projection back to patches
        self.head = nn.Linear(dim, patch * patch * in_ch)

    def forward(self, x, t, y):
        # x: B,C,H,W normalized to [-1,1], t in [0,1], y in [0..num_classes]
        b, c, h, w = x.shape
        seq, (ph, pw) = self.patch_embed(x)  # B,N,D
        seq = self.pos(seq)

        t_tok = self.time_mlp(self.time_sin(t)).unsqueeze(1)
        y_tok = self.label_emb(y).unsqueeze(1)

        cls = self.cls_token.expand(b, -1, -1)
        tt = self.time_token.expand(b, -1, -1) + t_tok
        lt = self.label_token.expand(b, -1, -1) + y_tok

        tokens = torch.cat([cls, tt, lt, seq], dim=1)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        seq_out = tokens[:, 3:, :]  # drop special tokens
        patches = self.head(seq_out)  # B,N,patch*patch*C
        patches = patches.view(b, ph * pw, c, self.patch, self.patch)
        # fold back
        patches = patches.permute(0, 2, 1, 3, 4).contiguous()  # B,C,N,ph,pw
        patches = patches.view(b, c, ph, pw, self.patch, self.patch)
        patches = patches.permute(0, 1, 2, 4, 3, 5).contiguous()
        x_recon = patches.view(b, c, ph * self.patch, pw * self.patch)
        return x_recon  # interpret as epsilon prediction

    @torch.no_grad()
    def sample(self, shape, betas, y, cfg_scale: float = 0.0, device: Optional[torch.device] = None, eta: float = 0.0):
        device = device or next(self.parameters()).device
        b = shape[0]
        x = torch.randn(shape, device=device)
        alphas = 1.0 - betas
        ac = torch.cumprod(alphas, dim=0)
        sqrt_ac = torch.sqrt(ac)
        sqrt_om = torch.sqrt(1 - ac)
        null = torch.full_like(y, self.num_classes)

        for i in reversed(range(len(betas))):
            t = torch.full((b,), (i + 0.5) / len(betas), device=device)
            eps_c = self.forward(x, t, y)
            if cfg_scale > 0:
                eps_u = self.forward(x, t, null)
                eps = eps_u + cfg_scale * (eps_c - eps_u)
            else:
                eps = eps_c
            a = ac[i]
            x0 = (x - sqrt_om[i] * eps) / (sqrt_ac[i] + 1e-8)
            if i == 0:
                x = x0
            else:
                a_prev = ac[i - 1]
                sigma = eta * math.sqrt((1 - a_prev) / (1 - a) * (1 - a / a_prev))
                noise = torch.randn_like(x) if sigma > 0 else 0.0
                x = torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev - sigma ** 2) * eps + sigma * noise
        return x.clamp(-1, 1)
