import math
from typing import Tuple

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    def __init__(self, img_size: int = 32, patch_size: int = 4, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # (B, C, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, C)
        return x


class TransformerEncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x


class MAEViT(nn.Module):
    """
    Minimal masked autoencoder (ViT-style) for CIFAR-like images.

    - Patches the image.
    - Applies a stack of Transformer encoder blocks to visible tokens.
    - Uses a lightweight decoder (two linear layers) to reconstruct pixel
      patches from token embeddings.

    Masking is driven by the training code; this module accepts a sequence of
    tokens and reconstructs all patches.
    """

    def __init__(
        self,
        img_size: int = 32,
        patch_size: int = 4,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 4,
        num_heads: int = 3,
    ):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_heads = num_heads

        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches

        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        blocks = []
        for _ in range(depth):
            blocks.append(TransformerEncoderBlock(embed_dim, num_heads))
        self.blocks = nn.ModuleList(blocks)
        self.norm = nn.LayerNorm(embed_dim)

        patch_dim = in_chans * (patch_size**2)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, patch_dim),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def encode_patches(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, C, H, W)
        returns: token embeddings (B, N, D)
        """
        tokens = self.patch_embed(x)
        tokens = tokens + self.pos_embed
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens

    def decode_patches(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        tokens: (B, N, D)
        returns: reconstructed image (B, C, H, W)
        """
        B, N, _ = tokens.shape
        patch_dim = self.in_chans * (self.patch_size**2)
        patches = self.decoder(tokens)  # (B, N, patch_dim)
        patches = patches.view(B, N, self.in_chans, self.patch_size, self.patch_size)

        h_patches = self.img_size // self.patch_size
        patches = patches.view(
            B, h_patches, h_patches, self.in_chans, self.patch_size, self.patch_size
        )
        patches = patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        img = patches.view(B, self.in_chans, self.img_size, self.img_size)
        return img

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        For STL training we do simple random masking inside the model for now:
        we drop a fraction of patch tokens and ask the decoder to reconstruct
        the full image.
        """
        tokens = self.encode_patches(x)
        x_rec = self.decode_patches(tokens)
        return x_rec, tokens


def mae_total_neurons(embed_dim: int, depth: int) -> int:
    """
    Capacity proxy: embed_dim * depth.
    """
    return int(embed_dim * depth)

