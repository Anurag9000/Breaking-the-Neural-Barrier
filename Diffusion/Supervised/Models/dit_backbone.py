import torch
import torch.nn as nn
from core.diffusion_core import SinusoidalTimeEmbedding


class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, patch=4, dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch, stride=patch)

    def forward(self, x):
        x = self.proj(x)
        b, c, h, w = x.shape
        # Flatten spatial dimensions and transpose to (B, N, C)
        return x.flatten(2).transpose(1, 2), (h, w)


class DiTBackbone(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=256, depth=8, heads=8, patch=4, cond_dim=0):
        super().__init__()
        self.embed = PatchEmbed(in_ch, patch, dim)
        self.pos = nn.Parameter(torch.randn(1, 4096, dim))  # max tokens = 64*64 / patch^2 = 4096 if patch=1
        self.time = SinusoidalTimeEmbedding(dim)
        self.cproj = nn.Linear(cond_dim, dim) if cond_dim > 0 else None

        layer = nn.TransformerEncoderLayer(d_model=dim, nhead=heads, batch_first=True)
        self.enc = nn.TransformerEncoder(layer, num_layers=depth)

        self.out = nn.Linear(dim, dim)
        self.unpatch = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, x, t, cond=None):
        # Embed patches
        tok, (h, w) = self.embed(x)
        n = tok.size(1)

        # Time embedding
        t_emb = self.time(t)
        if self.cproj is not None and cond is not None:
            t_emb = t_emb + self.cproj(cond)

        # Add positional and time embeddings
        tok = tok + t_emb.unsqueeze(1) + self.pos[:, :n]

        # Transformer encoding
        tok = self.enc(tok)
        tok = self.out(tok)

        # Reshape back to image grid
        b = tok.size(0)
        tok = tok.transpose(1, 2).reshape(b, -1, h, w)

        # Reconstruct image
        return self.unpatch(tok)
