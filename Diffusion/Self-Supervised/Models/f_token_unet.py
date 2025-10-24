# models/f_token_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# FiLM modulation
# -----------------------
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


# -----------------------
# Residual Block with FiLM
# -----------------------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.gn1 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
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


# -----------------------
# Downsampling block
# -----------------------
class Down(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)
        self.down = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)

    def forward(self, x, c):
        x = self.block(x, c)
        return self.down(x)


# -----------------------
# Upsampling block
# -----------------------
class Up(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)

    def forward(self, x, skip, c):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.block(x, c)


# -----------------------
# Time embedding helper
# -----------------------
def time_embedding(t, dim):
    """Simple sinusoidal embedding"""
    device = t.device
    half_dim = dim // 2
    emb = torch.exp(torch.arange(half_dim, device=device) * -(math.log(10000) / half_dim))
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


# -----------------------
# Token UNet
# -----------------------
class TokenUNet(nn.Module):
    """
    UNet over token embeddings.
    Input/output channels are embedding width.
    Final head maps to K logits per channel.
    """
    def __init__(self, embed_dim=64, base=64, channels=3, K=16, time_dim=256):
        super().__init__()
        self.channels = channels
        self.K = K
        self.time_dim = time_dim

        # Time conditioning
        self.cond = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        # Encoder
        self.enc1 = ResBlock(embed_dim, base, time_dim)
        self.down1 = Down(base, base * 2, time_dim)
        self.down2 = Down(base * 2, base * 4, time_dim)

        # Bottleneck
        self.mid = ResBlock(base * 4, base * 4, time_dim)

        # Decoder
        self.up1 = Up(base * 8, base * 2, time_dim)
        self.up2 = Up(base * 4, base, time_dim)

        # Output
        self.out = nn.Conv2d(base * 2, channels * K, 3, padding=1)

    def forward(self, x_emb, t):
        c = self.cond(time_embedding(t, self.time_dim))
        s1 = self.enc1(x_emb, c)
        x = self.down1(s1, c)
        s2 = x
        x = self.down2(x, c)
        x = self.mid(x, c)
        x = self.up1(x, s2, c)
        x = self.up2(x, s1, c)
        return self.out(x)  # [B, C*K, H, W]
