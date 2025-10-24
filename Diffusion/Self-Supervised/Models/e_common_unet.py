# models/e_common_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F


def sinusoidal_time_embedding(t, dim):
    """
    Create a sinusoidal time embedding.
    """
    half_dim = dim // 2
    emb = torch.exp(
        torch.arange(half_dim, dtype=torch.float32, device=t.device)
        * -(math.log(10000) / (half_dim - 1))
    )
    emb = t[:, None] * emb[None, :]
    emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)
    return emb


class FiLM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(in_dim, out_dim * 2))

    def forward(self, t_emb: torch.Tensor):
        h = self.mlp(t_emb)
        scale, shift = h.chunk(2, dim=1)
        return scale, shift


class ResBlock(nn.Module):
    def __init__(self, c_in: int, c_out: int, t_dim: int):
        super().__init__()
        self.conv1 = nn.Conv2d(c_in, c_out, 3, padding=1)
        self.gn1 = nn.GroupNorm(8, c_out)
        self.conv2 = nn.Conv2d(c_out, c_out, 3, padding=1)
        self.gn2 = nn.GroupNorm(8, c_out)
        self.film = FiLM(t_dim, c_out)
        self.skip = nn.Identity() if c_in == c_out else nn.Conv2d(c_in, c_out, 1)

    def forward(self, x, t_emb):
        s, b = self.film(t_emb)
        h = self.conv1(x)
        h = self.gn1(h)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.gn2(h)
        h = h * (1 + s[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(h)
        return h + self.skip(x)


class Down(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, t_dim)
        self.pool = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)

    def forward(self, x, t_emb):
        x = self.block(x, t_emb)
        return self.pool(x)


class Up(nn.Module):
    def __init__(self, c_in, c_out, t_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, t_dim)

    def forward(self, x, skip, t_emb):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.block(x, t_emb)


class UNetE(nn.Module):
    def __init__(self, base=64, time_dim=256, in_ch=3, out_ch=3):
        super().__init__()
        self.time_dim = time_dim
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )

        self.enc1 = ResBlock(in_ch, base, time_dim)
        self.down1 = Down(base, base * 2, time_dim)
        self.down2 = Down(base * 2, base * 4, time_dim)
        self.mid = ResBlock(base * 4, base * 4, time_dim)
        self.up1 = Up(base * 8, base * 2, time_dim)
        self.up2 = Up(base * 4, base, time_dim)
        self.out = nn.Conv2d(base * 2, out_ch, 3, padding=1)

    def forward(self, x, t):
        if t.dim() == 2:
            t = t.squeeze(1)
        t_emb = sinusoidal_time_embedding(t, self.time_dim)
        t_emb = self.time_mlp(t_emb)

        s1 = self.enc1(x, t_emb)
        x = self.down1(s1, t_emb)
        s2 = x
        x = self.down2(x, t_emb)
        x = self.mid(x, t_emb)
        x = self.up1(x, s2, t_emb)
        x = self.up2(x, s1, t_emb)
        return self.out(x)
