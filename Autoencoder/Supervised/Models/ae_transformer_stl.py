import math
import torch
import torch.nn as nn
from typing import Tuple

# -----------------------------------------------------------------------------
# AE_TRANSFORMER_STL: ViT-style patch autoencoder (single model)
# - Patchify 32x32 into (N= (32/ps)^2) tokens via conv-proj.
# - Encoder: Transformer encoder blocks.
# - Decoder: MLP head to predict pixel patches, then unpatchify to image.
# -----------------------------------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=192, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.patch_size = patch_size
    def forward(self, x):
        x = self.proj(x)  # (B, D, H/ps, W/ps)
        B, D, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x, (H, W)

class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)),
            nn.GELU(),
            nn.Linear(int(dim*mlp_ratio), dim),
        )
    def forward(self, x):
        h = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)[0]
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x

class AE_TRANSFORMER_STL(nn.Module):
    def __init__(self, in_channels: int = 3, embed_dim: int = 192, depth: int = 6,
                 num_heads: int = 6, patch_size: int = 4, mlp_ratio: float = 4.0):
        super().__init__()
        self.patch = PatchEmbed(in_ch=in_channels, embed_dim=embed_dim, patch_size=patch_size)
        self.pos = None  # learned pos embed optional; we'll use sin-cos for simplicity
        self.blocks = nn.ModuleList([TransformerBlock(embed_dim, num_heads, mlp_ratio) for _ in range(depth)])
        # Decoder: predict raw pixel patches from token embeddings
        self.head = nn.Linear(embed_dim, in_channels * patch_size * patch_size)

    @staticmethod
    def _build_2d_sincos_pos_embed(h: int, w: int, dim: int, device):
        y, x = torch.meshgrid(torch.arange(h, device=device), torch.arange(w, device=device), indexing='ij')
        assert dim % 4 == 0
        dim_half = dim // 2
        omega = torch.arange(dim_half//2, device=device).float()
        omega = 1. / (10000 ** (omega / (dim_half//2)))
        out = []
        for grid in (x, y):
            emb = torch.einsum('hw,c->hwc', grid.float(), omega)
            out += [torch.sin(emb), torch.cos(emb)]
        pos = torch.cat(out, dim=2).reshape(1, h*w, dim)
        return pos

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int,int]]:
        tokens, (H, W) = self.patch(x)
        if self.pos is None or self.pos.shape[1] != H*W:
            self.pos = self._build_2d_sincos_pos_embed(H, W, tokens.size(-1), tokens.device)
        h = tokens + self.pos
        for blk in self.blocks:
            h = blk(h)
        return h, (H, W)

    def decode(self, h: torch.Tensor, hw: Tuple[int,int]) -> torch.Tensor:
        B, N, D = h.shape
        H, W = hw
        patches = self.head(h)  # (B,N,C*ps*ps)
        C = 3; ps = 4
        patches = patches.view(B, H, W, C, ps, ps).permute(0,3,1,4,2,5).contiguous()
        x_rec = patches.view(B, C, H*ps, W*ps)
        return x_rec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h, hw = self.encode(x)
        x_rec = self.decode(h, hw)
        return x_rec, h


def ae_transformer_total_neurons(embed_dim: int, depth: int) -> int:
    return int(embed_dim * (depth + 1))
