# models/sde_common_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------
# Sinusoidal Embedding
# ----------------------------
def sinusoidal_embed(args, dim):
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# ----------------------------
# FiLM Conditioning Layer
# ----------------------------
class FiLM(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_dim * 2)
        )

    def forward(self, h):
        a, b = self.mlp(h).chunk(2, dim=1)
        return a, b


# ----------------------------
# Residual Block
# ----------------------------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.gn1 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, c_out)
        self.film = FiLM(cond_dim, c_out)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x, cond):
        a, b = self.film(cond)
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = h * (1 + a[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(h)
        return h + self.skip(x)


# ----------------------------
# Downsampling Block
# ----------------------------
class Down(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)
        self.down = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)

    def forward(self, x, c):
        x = self.block(x, c)
        return self.down(x)


# ----------------------------
# Upsampling Block
# ----------------------------
class Up(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)

    def forward(self, x, skip, c):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.block(x, c)


# ----------------------------
# Score U-Net for SDE Models
# ----------------------------
class ScoreUNetSDE(nn.Module):
    def __init__(self, base=64, in_ch=3, out_ch=3, cond_dim=256):
        super().__init__()
        self.cond_dim = cond_dim

        self.cond = nn.Sequential(
            nn.Linear(cond_dim, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        self.enc1 = ResBlock(in_ch, base, cond_dim)
        self.down1 = Down(base, base * 2, cond_dim)
        self.down2 = Down(base * 2, base * 4, cond_dim)
        self.mid = ResBlock(base * 4, base * 4, cond_dim)
        self.up1 = Up(base * 8, base * 2, cond_dim)
        self.up2 = Up(base * 4, base, cond_dim)
        self.out = nn.Conv2d(base * 2, out_ch, 3, padding=1)

    def forward(self, x, t):
        c = self.cond(sinusoidal_embed(t, self.cond_dim))
        s1 = self.enc1(x, c)
        x = self.down1(s1, c)
        s2 = x
        x = self.down2(x, c)
        x = self.mid(x, c)
        x = self.up1(x, s2, c)
        x = self.up2(x, s1, c)
        return self.out(x)
