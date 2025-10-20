import math
from typing import Optional
import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=256):
        super().__init__()
        self.grid = (img_size // patch_size, img_size // patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        # x: (B, C, H, W) -> (B, N, D)
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        return x

class ViT(nn.Module):
    """Vision Transformer (Dosovitskiy et al.) for image classification."""
    def __init__(self, num_classes: int = 10, img_size: int = 32, patch_size: int = 4,
                 in_chans: int = 3, embed_dim: int = 256, depth: int = 6, nhead: int = 8,
                 mlp_ratio: float = 4.0, dropout: float = 0.1):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + (img_size // patch_size)**2, embed_dim))
        enc_layer = nn.TransformerEncoderLayer(embed_dim, nhead, int(embed_dim*mlp_ratio), dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        nn.init.xavier_uniform_(self.head.weight); nn.init.zeros_(self.head.bias)

    def forward(self, x):
        x = self.patch(x)
        B, N, D = x.size()
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1) + self.pos[:, : N+1]
        x = self.encoder(x)
        x = self.norm(x[:, 0])
        return self.head(x)
