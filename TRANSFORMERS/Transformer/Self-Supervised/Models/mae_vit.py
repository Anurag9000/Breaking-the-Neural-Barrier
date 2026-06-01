"""
MAE-ViT (single-model, no teacher)
- ViT encoder on visible patches only
- Lightweight decoder reconstructs pixels of masked patches
- Loss: MSE over masked patches (normalized)
Style: similar to uploaded ADP CNN files (PyTorch, early-stop friendly)
"""
from __future__ import annotations
import math
from dataclasses import dataclass
from typing import Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Minimal ViT backbone
# ------------------------------
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        assert img_size % patch_size == 0
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.num_patches = self.grid * self.grid
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (B, C, H, W)
        x = self.proj(x)  # (B, E, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, E)
        return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads=6, mlp_ratio=4.0, drop=0.0, attn_drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(embed_dim=dim, num_heads=num_heads, dropout=attn_drop, batch_first=True)
        self.drop_path = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size=32, patch_size=4, embed_dim=384, depth=6, num_heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, 3, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.patch_embed.num_patches, embed_dim))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, drop) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x, token_indices: Optional[torch.Tensor] = None):
        # if token_indices is provided, we select a subset of tokens (visible tokens for MAE)
        x = self.patch_embed(x)  # (B, N, E)
        x = x + self.pos_embed
        if token_indices is not None:
            # token_indices: (B, n_vis) with indices into N
            B = x.size(0)
            batch_indices = torch.arange(B, device=x.device).unsqueeze(-1)
            x = x[batch_indices, token_indices]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x  # (B, n_vis or N, E)

# ------------------------------
# MAE model
# ------------------------------
@dataclass
class MAEConfig:
    img_size: int = 32
    patch_size: int = 4
    embed_dim: int = 384
    depth: int = 6
    num_heads: int = 6
    mlp_ratio: float = 4.0
    decoder_dim: int = 192
    decoder_depth: int = 4
    decoder_heads: int = 3
    mask_ratio: float = 0.6

class MAEViT(nn.Module):
    def __init__(self, cfg: MAEConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ViTEncoder(cfg.img_size, cfg.patch_size, cfg.embed_dim, cfg.depth, cfg.num_heads, cfg.mlp_ratio)
        # decoder sees full set of tokens after we re-insert masked tokens as learned mask tokens
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.decoder_dim))
        self.enc_to_dec = nn.Linear(cfg.embed_dim, cfg.decoder_dim)
        self.dec_pos = nn.Parameter(torch.zeros(1, (cfg.img_size//cfg.patch_size)**2, cfg.decoder_dim))
        self.decoder_blocks = nn.ModuleList([
            Block(cfg.decoder_dim, cfg.decoder_heads, 4.0) for _ in range(cfg.decoder_depth)
        ])
        self.decoder_norm = nn.LayerNorm(cfg.decoder_dim)
        # predict pixels per patch
        self.patch_pixels = (cfg.patch_size ** 2) * 3
        self.head = nn.Linear(cfg.decoder_dim, self.patch_pixels)
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        nn.init.trunc_normal_(self.mask_token, std=0.02)

    def random_mask(self, B: int, N: int, device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return visible_indices, masked_indices, restore_indices."""
        n_mask = int(self.cfg.mask_ratio * N)
        idx = torch.rand(B, N, device=device).argsort(dim=1)
        masked = idx[:, :n_mask]
        visible = idx[:, n_mask:]
        # restore indices map to original token order
        restore = torch.argsort(idx, dim=1)
        return visible, masked, restore

    def forward(self, imgs):
        B = imgs.size(0)
        N = (self.cfg.img_size // self.cfg.patch_size) ** 2
        visible, masked, restore = self.random_mask(B, N, imgs.device)
        # encode only visible tokens
        enc_tokens = self.encoder(imgs, token_indices=visible)  # (B, n_vis, E)
        dec_tokens = self.enc_to_dec(enc_tokens)
        # re-insert mask tokens to get full N tokens
        mask_tok = self.mask_token.expand(B, masked.size(1), -1)
        # concatenate and then unshuffle to original order
        full_tokens = torch.cat([dec_tokens, mask_tok], dim=1)  # (B, n_vis+n_mask, D)
        # Build indices that place tokens back: gather on concat([visible, masked]) order
        gather_idx = torch.cat([visible, masked], dim=1)
        # reorder full_tokens to match original token positions
        B_idx = torch.arange(B, device=imgs.device).unsqueeze(-1).expand_as(gather_idx)
        full_ordered = torch.zeros(B, N, full_tokens.size(-1), device=imgs.device)
        full_ordered[B_idx, gather_idx] = full_tokens
        # add decoder pos embed
        x = full_ordered + self.dec_pos
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)
        pred = self.head(x)  # (B, N, patch_pixels)
        # target: patchify images
        target = self.patchify(imgs)
        # compute MSE over masked patches only
        B_idx2 = torch.arange(B, device=imgs.device).unsqueeze(-1)
        masked_pred = pred[B_idx2, masked]
        masked_tgt = target[B_idx2, masked]
        loss = F.mse_loss(masked_pred, masked_tgt)
        return loss, {"masked_mse": loss.item(), "n_mask": masked.size(1)}

    def patchify(self, imgs):
        p = self.cfg.patch_size
        B, C, H, W = imgs.shape
        assert H == W == self.cfg.img_size
        x = imgs.reshape(B, C, H//p, p, W//p, p)
        x = x.permute(0,2,4,3,5,1).reshape(B, (H//p)*(W//p), p*p*C)
        return x
