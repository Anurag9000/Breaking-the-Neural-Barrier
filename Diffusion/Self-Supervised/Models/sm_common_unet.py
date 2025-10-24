# models/sm_common_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------
# FiLM layer
# -------------------------
class FiLM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_dim * 2)
        )

    def forward(self, cond):
        h = self.mlp(cond)
        scale, shift = h.chunk(2, dim=1)
        return scale, shift


# -------------------------
# Residual Block
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim, groups=8):
        super().__init__()
        self.c_in, self.c_out = c_in, c_out
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.gn1 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, c_out)
        self.film = FiLM(cond_dim, c_out)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x, cond):
        s, b = self.film(cond)
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = h * (1 + s[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(h)
        return h + self.skip(x)


# -------------------------
# Downsample block
# -------------------------
class Down(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)
        self.down = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)

    def forward(self, x, cond):
        x = self.block(x, cond)
        return self.down(x)


# -------------------------
# Upsample block
# -------------------------
class Up(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)

    def forward(self, x, skip, cond):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.block(x, cond)


# -------------------------
# UNet
# -------------------------
class ScoreUNet(nn.Module):
    """Unconditional score network; outputs ∇_x log q_σ(x)."""
    def __init__(self, base=64, cond_dim=256, in_ch=3, out_ch=3):
        super().__init__()
        self.cond_dim = cond_dim
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        # Encoder
        self.enc1 = ResBlock(in_ch, base, cond_dim)
        self.down1 = Down(base, base*2, cond_dim)
        self.down2 = Down(base*2, base*4, cond_dim)

        # Mid
        self.mid = ResBlock(base*4, base*4, cond_dim)

        # Decoder
        self.up1 = Up(base*4 + base*4, base*2, cond_dim)
        self.up2 = Up(base*2 + base*2, base, cond_dim)
        self.out = nn.Conv2d(base + base, out_ch, 3, padding=1)

    def forward(self, x, sigmas):
        cond = sigma_embedding(sigmas, self.cond_dim)
        cond = self.cond_mlp(cond)

        s1 = self.enc1(x, cond)
        x = self.down1(s1, cond)
        s2 = x
        x = self.down2(x, cond)
        x = self.mid(x, cond)
        x = self.up1(x, s2, cond)
        x = self.up2(x, s1, cond)
        return self.out(x)
