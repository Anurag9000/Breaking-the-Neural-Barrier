import torch
import torch.nn as nn
import torch.nn.functional as F


# ====== Sinusoidal Time Embedding ======
class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.fc = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, dim)
        )

    def forward(self, t):
        half = self.dim // 2
        freqs = torch.exp(
            torch.arange(half, device=t.device)
            * (-torch.log(torch.tensor(10000.0)) / (half - 1))
        )
        ang = t.float().unsqueeze(1) * freqs.unsqueeze(0)
        emb = torch.cat([torch.sin(ang), torch.cos(ang)], dim=1)

        if emb.shape[1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[1]))

        return self.fc(emb)


# ====== Residual Block ======
class ResBlock(nn.Module):
    def __init__(self, c, tdim, c_out=None):
        super().__init__()
        self.c_out = c_out or c
        self.conv1 = nn.Conv2d(c, self.c_out, 3, padding=1)
        self.conv2 = nn.Conv2d(self.c_out, self.c_out, 3, padding=1)
        self.time = nn.Sequential(nn.SiLU(), nn.Linear(tdim, self.c_out))
        self.skip = nn.Conv2d(c, self.c_out, 1) if self.c_out != c else nn.Identity()
        self.g1 = nn.GroupNorm(8, self.c_out)
        self.g2 = nn.GroupNorm(8, self.c_out)

    def forward(self, x, t_emb):
        h = self.conv1(x)
        h = self.g1(h) + self.time(t_emb).unsqueeze(-1).unsqueeze(-1)
        h = F.silu(h)
        h = self.conv2(h)
        h = self.g2(h)
        return F.silu(h + self.skip(x))


# ====== Score U-Net ======
class ScoreUNet(nn.Module):
    """Predict score field with same shape as input x."""

    def __init__(self, img_ch=3, base=64, tdim=256, cond_dim=0):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)
        self.cproj = nn.Linear(cond_dim, tdim) if cond_dim > 0 else None

        self.inp = nn.Conv2d(img_ch, base, 3, padding=1)
        self.d1 = ResBlock(base, tdim, base)
        self.d2 = ResBlock(base, tdim, base * 2)
        self.d3 = ResBlock(base * 2, tdim, base * 4)
        self.mid = ResBlock(base * 4, tdim)
        self.u3 = ResBlock(base * 4, tdim, base * 2)
        self.u2 = ResBlock(base * 2, tdim, base)
        self.u1 = ResBlock(base, tdim, base)
        self.out = nn.Conv2d(base, img_ch, 3, padding=1)

    def forward(self, x, t, cond=None):
        t_emb = self.time(t)
        if self.cproj is not None and cond is not None:
            t_emb = t_emb + self.cproj(cond)

        x0 = self.inp(x)
        d1 = self.d1(x0, t_emb)
        x1 = F.avg_pool2d(d1, 2)
        d2 = self.d2(x1, t_emb)
        x2 = F.avg_pool2d(d2, 2)
        d3 = self.d3(x2, t_emb)

        m = self.mid(F.avg_pool2d(d3, 2), t_emb)
        u3 = F.interpolate(m, scale_factor=2, mode='nearest')
        u3 = self.u3(u3 + d3, t_emb)
        u2 = F.interpolate(u3, scale_factor=2, mode='nearest')
        u2 = self.u2(u2 + d2, t_emb)
        u1 = F.interpolate(u2, scale_factor=2, mode='nearest')
        u1 = self.u1(u1 + d1, t_emb)

        return self.out(u1)
