# models/flow_common_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------
# Residual block (must exist above this snippet in your actual file)
# ---------------------------------------------------------------------
class ResBlock(nn.Module):
    def __init__(self, c_in, c_out, cond_dim, groups=8):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.gn1 = nn.GroupNorm(groups, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.gn2 = nn.GroupNorm(groups, c_out)
        self.film = nn.Sequential(
            nn.SiLU(),
            nn.Linear(cond_dim, c_out * 2)
        )
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x, cond):
        a, b = self.film(cond).chunk(2, 1)
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = h * (1 + a[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(h)
        return h + self.skip(x)


# ---------------------------------------------------------------------
# Downsample block
# ---------------------------------------------------------------------
class Down(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)
        self.down = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)

    def forward(self, x, c):
        x = self.block(x, c)
        return self.down(x)


# ---------------------------------------------------------------------
# Upsample block
# ---------------------------------------------------------------------
class Up(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)

    def forward(self, x, skip, c):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.block(x, c)


# ---------------------------------------------------------------------
# Velocity UNet (core architecture)
# ---------------------------------------------------------------------
class VelocityUNet(nn.Module):
    def __init__(self, base=64, in_ch=3, out_ch=3, cond_dim=256, use_null_token=False):
        super().__init__()
        self.cond_dim = cond_dim
        self.use_null = use_null_token

        extra = 16 if use_null_token else 0
        self.cond_mlp = nn.Sequential(
            nn.Linear(cond_dim + extra, cond_dim),
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

        if use_null_token:
            self.null_embed = nn.Parameter(torch.zeros(1, 16))
            nn.init.normal_(self.null_embed, std=0.02)

    def forward(self, x, t, use_null=False):
        cond = time_embedding(t, self.cond_dim)

        if self.use_null:
            tok = self.null_embed.expand(cond.size(0), -1)
            if use_null:
                cond = torch.cat([cond, tok], dim=1)
            else:
                cond = torch.cat([cond, torch.zeros_like(tok)], dim=1)

        cond = self.cond_mlp(cond)

        s1 = self.enc1(x, cond)
        x = self.down1(s1, cond)
        s2 = x
        x = self.down2(x, cond)
        x = self.mid(x, cond)
        x = self.up1(x, s2, cond)
        x = self.up2(x, s1, cond)
        return self.out(x)
