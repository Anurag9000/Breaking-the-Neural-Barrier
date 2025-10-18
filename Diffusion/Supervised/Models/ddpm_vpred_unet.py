import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse UNet blocks for v-prediction variant
from ddpm_classcond_unet import SinusoidalTimeEmbedding, MLPTime, ResBlock, AttnBlock, Down, Up


class DDPMVPredUNet(nn.Module):
    """U-Net that predicts v (velocity) for DDPM.
    v = alpha^{1/2} * eps - (1 - alpha)^{1/2} * x0 ; stable with cosine schedules.
    We implement the head identically to the epsilon model but interpret outputs as v.
    """
    def __init__(self, in_ch: int = 3, base_ch: int = 64, ch_mults=(1, 2, 2, 4),
                 num_classes: int = 10, time_dim: int = 256):
        super().__init__()
        self.in_ch = in_ch
        self.time_emb = SinusoidalTimeEmbedding(time_dim)
        self.time_mlp = MLPTime(time_dim, time_dim)
        self.label_emb = nn.Embedding(num_classes + 1, time_dim)  # +1 null

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        self.downs = nn.ModuleList()
        ch = base_ch
        self.skips_ch = []
        for m in ch_mults:
            out_ch = base_ch * m
            self.downs.append(Down(ch, out_ch, time_dim, use_attn=False))
            self.skips_ch.append(out_ch)
            ch = out_ch

        self.mid1 = ResBlock(ch, ch, time_dim)
        self.mid_attn = AttnBlock(ch)
        self.mid2 = ResBlock(ch, ch, time_dim)

        self.ups = nn.ModuleList()
        for m, skip_ch in zip(reversed(ch_mults), reversed(self.skips_ch)):
            out_ch = base_ch * m
            self.ups.append(Up(ch, out_ch, time_dim, use_attn=False))
            ch = out_ch

        self.out_norm = nn.GroupNorm(8, ch)
        self.out_conv = nn.Conv2d(ch, in_ch, 3, padding=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        t_feat = self.time_mlp(self.time_emb(t)) + self.label_emb(y)
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
        v = self.out_conv(x)
        return v

    @torch.no_grad()
    def sample(self, shape, alphas_cumprod: torch.Tensor, y: torch.Tensor, cfg_scale: float = 0.0,
               device: Optional[torch.device] = None):
        device = device or next(self.parameters()).device
        b = shape[0]
        x = torch.randn(shape, device=device)
        null_label = torch.full_like(y, self.label_emb.num_embeddings - 1)

        for i in reversed(range(len(alphas_cumprod))):
            t = torch.full((b,), (i + 0.5) / len(alphas_cumprod), device=device)
            a = alphas_cumprod[i]
            sqrt_a = torch.sqrt(a)
            sqrtm = torch.sqrt(1 - a)
            v_cond = self.forward(x, t, y)
            if cfg_scale > 0:
                v_null = self.forward(x, t, null_label)
                v = v_null + cfg_scale * (v_cond - v_null)
            else:
                v = v_cond
            # recover eps and x0
            eps = (v + sqrtm * 0 - 0) / sqrt_a  # v = sqrt(a)*eps - sqrt(1-a)*x0; but we don't know x0 here
            # Use direct update via v-formula from Stable Diffusion's v-parameterization derivation:
            x0 = sqrt_a * x - sqrtm * v
            if i == 0:
                x = x0
            else:
                a_prev = alphas_cumprod[i - 1]
                noise = torch.randn_like(x)
                x = torch.sqrt(a_prev) * x0 + torch.sqrt(1 - a_prev) * noise
        return x.clamp(-1, 1)
