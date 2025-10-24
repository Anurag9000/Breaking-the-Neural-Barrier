import torch
import torch.nn as nn
import torch.nn.functional as F
from core.diffusion_core import SinusoidalTimeEmbedding


class ConvStem(nn.Module):
    def __init__(self, in_ch=3, c=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.SiLU()
        )

    def forward(self, x):
        return self.net(x)


class TinyTransformer(nn.Module):
    def __init__(self, dim=256, depth=4, heads=8):
        super().__init__()
        lyr = nn.TransformerEncoderLayer(dim, heads, batch_first=True)
        self.enc = nn.TransformerEncoder(lyr, num_layers=depth)

    def forward(self, tok):
        return self.enc(tok)


class HybridCTDiff(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, base=64, dim=256, patch=4, tdim=256, cond_dim=0):
        super().__init__()
        self.time = SinusoidalTimeEmbedding(tdim)
        self.cproj = nn.Linear(cond_dim, tdim) if cond_dim > 0 else None

        self.stem = ConvStem(in_ch, base)
        self.patch = nn.Conv2d(base, dim, kernel_size=patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, 4096, dim))  # max 4096 patches
        self.tr = TinyTransformer(dim)
        self.up = nn.ConvTranspose2d(dim, base, kernel_size=patch, stride=patch)
        self.head = nn.Conv2d(base, out_ch, 3, padding=1)

    def forward(self, x, t, cond=None):
        t_emb = self.time(t)
        if self.cproj is not None and cond is not None:
            t_emb = t_emb + self.cproj(cond)

        # Stem convolution
        h = self.stem(x) + t_emb.unsqueeze(-1).unsqueeze(-1)

        # Patch embedding
        tok = self.patch(h)
        b, c, hh, ww = tok.shape
        tok = tok.flatten(2).transpose(1, 2)  # (B, N_patches, dim)
        tok = tok + self.pos[:, :tok.size(1), :]  # add positional encoding

        # Transformer
        tok = self.tr(tok)

        # Reshape back to image
        tok = tok.transpose(1, 2).reshape(b, -1, hh, ww)
        h = self.up(tok)
        return self.head(h)
