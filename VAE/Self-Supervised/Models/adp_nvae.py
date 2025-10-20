from dataclasses import dataclass
from typing import Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class NVAEConfig:
    in_channels: int = 3
    img_size: int = 32
    z_levels: int = 3           # hierarchical latents
    z_dim: int = 32             # per-level latent size
    width: int = 128            # base channels

# Simple NVAE-style blocks (very lightweight approximation)
class ResBlock(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, ch)
        self.conv1 = nn.Conv2d(ch, ch, 3, 1, 1)
        self.norm2 = nn.GroupNorm(8, ch)
        self.conv2 = nn.Conv2d(ch, ch, 3, 1, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return x + h

class Down(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, 2, 1)
    def forward(self, x):
        return self.conv(x)

class Up(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.t = nn.ConvTranspose2d(ch, ch, 4, 2, 1)
    def forward(self, x):
        return self.t(x)

class EncStage(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.block = nn.Sequential(ResBlock(ch), ResBlock(ch))
    def forward(self, x):
        return self.block(x)

class NVAE(nn.Module):
    def __init__(self, cfg: NVAEConfig):
        super().__init__()
        self.cfg = cfg
        ch = cfg.width
        self.in_conv = nn.Conv2d(cfg.in_channels, ch, 3, 1, 1)
        # Multi-scale encoder
        self.enc_stages = nn.ModuleList([EncStage(ch) for _ in range(cfg.z_levels)])
        self.downs = nn.ModuleList([Down(ch) for _ in range(cfg.z_levels)])
        # Gaussian params per level (top-down posterior)
        self.to_mu = nn.ModuleList([nn.Conv2d(ch, cfg.z_dim, 1) for _ in range(cfg.z_levels)])
        self.to_lv = nn.ModuleList([nn.Conv2d(ch, cfg.z_dim, 1) for _ in range(cfg.z_levels)])
        # Decoder
        self.up = Up(ch)
        self.dec_blocks = nn.ModuleList([ResBlock(ch) for _ in range(cfg.z_levels)])
        self.prior_mu = nn.ModuleList([nn.Conv2d(ch, cfg.z_dim, 1) for _ in range(cfg.z_levels)])
        self.prior_lv = nn.ModuleList([nn.Conv2d(ch, cfg.z_dim, 1) for _ in range(cfg.z_levels)])
        self.from_z = nn.ModuleList([nn.Conv2d(cfg.z_dim, ch, 1) for _ in range(cfg.z_levels)])
        self.out_conv = nn.Conv2d(ch, cfg.in_channels, 1)

    @staticmethod
    def reparameterize(mu, lv):
        std = (0.5 * lv).exp()
        return mu + torch.randn_like(std) * std

    def forward(self, x):
        h = self.in_conv(x)
        enc_feats = []
        for i in range(self.cfg.z_levels):
            h = self.enc_stages[i](h)
            enc_feats.append(h)
            h = self.downs[i](h)
        # Top feature is h; now decode with hierarchical latents
        kl_total = 0.0
        out = None
        for i in reversed(range(self.cfg.z_levels)):
            # compute posterior params from encoder features
            mu_i = self.to_mu[i](enc_feats[i])
            lv_i = self.to_lv[i](enc_feats[i])
            # compute prior params from current decoder state (or zero if first)
            if out is None:
                p_mu = torch.zeros_like(mu_i)
                p_lv = torch.zeros_like(lv_i)
                dec_h = torch.zeros_like(enc_feats[i])
            else:
                p_mu = self.prior_mu[i](out)
                p_lv = self.prior_lv[i](out)
                dec_h = out
            z_i = self.reparameterize(mu_i, lv_i)
            # KL of diagonal Gaussians
            kl = 0.5 * ((lv_i.exp() / p_lv.exp()) + (mu_i - p_mu).pow(2) / p_lv.exp() - 1 + (p_lv - lv_i)).mean()
            kl_total = kl_total + kl
            # inject z
            dec_h = dec_h + self.from_z[i](z_i)
            dec_h = self.dec_blocks[i](dec_h)
            if i != 0:
                dec_h = self.up(dec_h)
            out = dec_h
        xr = torch.sigmoid(self.out_conv(out))
        return xr, kl_total

    def loss_fn(self, x, xr, kl) -> Dict[str, torch.Tensor]:
        recon = F.binary_cross_entropy(xr, x, reduction='mean')
        loss = recon + kl
        return {"loss": loss, "recon": recon, "kl": kl}
