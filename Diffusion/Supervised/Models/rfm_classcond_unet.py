import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional

# Reuse UNet blocks from the epsilon model for consistency
from ddpm_classcond_unet import SinusoidalTimeEmbedding, MLPTime, ResBlock, AttnBlock, Down, Up

class RectifiedFlowUNet(nn.Module):
    """Single-model Rectified Flow / Flow Matching denoiser (velocity field).
    We train on pairs (x0 ~ data, x1 ~ N(0,I)), sample t~U(0,1), x_t = (1-t)x0 + t x1.
    Target velocity v* = x1 - x0 (constant along straight path); predict v_theta(x_t, t, y).
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 64, ch_mults=(1, 2, 2, 4), num_classes: int = 10, time_dim: int = 256):
        super().__init__()
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = MLPTime(time_dim, time_dim)
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)
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
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        cond = self.time_mlp(self.time_emb(t)) + self.label_emb(y)
        h = self.in_conv(x)
        skips = []
        for d in self.downs:
            h, s = d(h, cond)
            skips.append(s)
        h = self.mid1(h, cond)
        h = self.mid_attn(h)
        h = self.mid2(h, cond)
        for u in self.ups:
            s = skips.pop()
            h = u(h, s, cond)
        h = F.silu(self.out_norm(h))
        v = self.out_conv(h)
        return v  # predicted velocity

    @torch.no_grad()
    def sample(self, x0_noise: torch.Tensor, steps: int, y: torch.Tensor, device: Optional[torch.device] = None):
        """Deterministic ODE integration along learned velocity field from t=1->0.
        Start from x1 ~ N(0,I) (given as x0_noise here for flexibility), integrate backward.
        Simple Euler/Heun integrator.
        """
        device = device or next(self.parameters()).device
        x = x0_noise.to(device)
        b = x.size(0)
        dt = 1.0 / steps
        null = torch.full_like(y, self.label_emb.num_embeddings - 1)
        for i in range(steps):
            t = torch.full((b,), 1.0 - i * dt, device=device)
            v = self.forward(x, t, y)
            x = x - dt * v  # reverse-time update for rectified flow
        return x.clamp(-1, 1)
