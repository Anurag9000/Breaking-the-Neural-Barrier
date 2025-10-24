# models/g_common_cunet.py
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# -------------------------
# Positional Encoding
# -------------------------
def pe(x, dim=256, device=None):
    """
    Standard sinusoidal positional encoding.
    Returns a tensor of shape (..., dim)
    """
    half = dim // 2
    freqs = torch.exp(torch.arange(half, device=device) * -(math.log(10000.0) / (half - 1)))
    angles = x.view(-1, 1) * freqs.view(1, -1)
    emb = torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def pair_embed(t, s, dim=256):
    """
    Encode pair of timesteps (t, s) and derived quantities.
    Returns concatenation of four positional encodings:
    - t, s, t-s, s/(t+eps)
    """
    eps = 1e-6
    z1 = pe(t, dim // 4)
    z2 = pe(s, dim // 4)
    z3 = pe((t - s).clamp(0, 1), dim // 4)
    z4 = pe((s / (t + eps)).clamp(0, 1), dim // 4)
    return torch.cat([z1, z2, z3, z4], dim=1)


# -------------------------
# Feature-wise Linear Modulation
# -------------------------
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


# -------------------------
# Residual Block
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, kernel_size=3, padding=1)
        self.gn1 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, kernel_size=3, padding=1)
        self.gn2 = nn.GroupNorm(groups, c_out)
        self.film = FiLM(cond_dim, c_out)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x, c):
        a, b = self.film(c)
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = h * (1 + a[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(h)
        return h + self.skip(x)


# -------------------------
# Downsample Block
# -------------------------
class Down(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)
        self.down = nn.Conv2d(c_out, c_out, kernel_size=3, stride=2, padding=1)

    def forward(self, x, c):
        x = self.block(x, c)
        return self.down(x)


# -------------------------
# Upsample Block
# -------------------------
class Up(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)

    def forward(self, x, skip, c):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.block(x, c)


# -------------------------
# Consistency UNet
# -------------------------
class ConsistencyUNet(nn.Module):
    def __init__(self, base=64, in_ch=3, out_ch=3, cond_dim=256):
        super().__init__()
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
        self.out = nn.Conv2d(base * 2, out_ch, kernel_size=3, padding=1)

    def forward(self, x_t, t, s):
        c = self.cond(pair_embed(t, s))
        s1 = self.enc1(x_t, c)
        x = self.down1(s1, c)
        s2 = x
        x = self.down2(x, c)
        x = self.mid(x, c)
        x = self.up1(x, s2, c)
        x = self.up2(x, s1, c)
        return self.out(x)
