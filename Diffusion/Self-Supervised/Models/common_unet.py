import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# Sinusoidal timestep embedding (like Transformer positional encoding)
def sinusoidal_time_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(
        torch.arange(half, device=device, dtype=torch.float32)
        * -(math.log(10000.0) / (half - 1))
    )
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0, 1))
    return emb


class FiLM(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(in_dim, out_dim * 2)
        )

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
        # FiLM modulation
        h = h * (1 + s[:, :, None, None]) + b[:, :, None, None]
        h = F.silu(h)
        return h + self.skip(x)
