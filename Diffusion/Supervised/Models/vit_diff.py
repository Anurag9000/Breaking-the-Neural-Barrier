import torch
import torch.nn as nn
from core.diffusion_core import SinusoidalTimeEmbedding


class ViTDiff(nn.Module):
    def __init__(self, in_ch=3, out_ch=3, dim=256, depth=6, heads=8, patch=4, cond_dim=0):
        super().__init__()
        self.patch = nn.Conv2d(in_ch, dim, patch, stride=patch)
        self.pos = nn.Parameter(torch.randn(1, 4096, dim))
        self.time = SinusoidalTimeEmbedding(dim)
        self.cproj = nn.Linear(cond_dim, dim) if cond_dim > 0 else None
        layer = nn.TransformerEncoderLayer(dim, heads, batch_first=True)
        self.vit = nn.TransformerEncoder(layer, num_layers=depth)
        self.out = nn.Linear(dim, dim)
        self.unpatch = nn.ConvTranspose2d(dim, out_ch, kernel_size=patch, stride=patch)

    def forward(self, x, t, cond=None):
        x = self.patch(x)
        b, c, h, w = x.shape
        tok = x.flatten(2).transpose(1, 2)

        t_emb = self.time(t)
        if self.cproj is not None and cond is not None:
            t_emb = t_emb + self.cproj(cond)

        tok = tok + t_emb.unsqueeze(1) + self.pos[:, :tok.size(1)]
        tok = self.vit(tok)
        tok = self.out(tok)
        tok = tok.transpose(1, 2).reshape(b, c, h, w)
        return self.unpatch(tok)
