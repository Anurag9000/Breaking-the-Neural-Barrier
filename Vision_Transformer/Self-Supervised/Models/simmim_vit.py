"""
SimMIM-ViT (single-model)
- Predict raw pixels for masked patches using a linear head (no heavy decoder)
- Loss: MSE on masked patches only
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# Reuse small ViT components (embedded here for standalone file)
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, embed_dim=384):
        super().__init__()
        assert img_size % patch_size == 0
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1, 2)
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
    def __init__(self, dim, heads=6, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0])
        x = x + self.drop(self.mlp(self.norm2(x)))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, dim=384, depth=6, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch_size, 3, dim)
        self.pos = nn.Parameter(torch.zeros(1, self.patch.num_patches, dim))
        self.blocks = nn.ModuleList([Block(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos, std=0.02)
    def forward(self, x):
        x = self.patch(x) + self.pos
        for blk in self.blocks: x = blk(x)
        return self.norm(x)

@dataclass
class SimMIMConfig:
    img_size:int=32; patch_size:int=4; embed_dim:int=384; depth:int=6; heads:int=6; mlp_ratio:float=4.0; mask_ratio:float=0.6

class SimMIMViT(nn.Module):
    def __init__(self, cfg: SimMIMConfig):
        super().__init__()
        self.cfg = cfg
        self.vit = ViT(cfg.img_size, cfg.patch_size, cfg.embed_dim, cfg.depth, cfg.heads, cfg.mlp_ratio)
        self.patch_pixels = (cfg.patch_size**2)*3
        self.pred_head = nn.Linear(cfg.embed_dim, self.patch_pixels)

    def random_mask(self, B, N, device):
        n_mask = int(self.cfg.mask_ratio * N)
        idx = torch.rand(B, N, device=device).argsort(dim=1)
        mask = idx[:, :n_mask]
        return mask

    def patchify(self, imgs):
        p = self.cfg.patch_size
        B, C, H, W = imgs.shape
        x = imgs.reshape(B, C, H//p, p, W//p, p).permute(0,2,4,3,5,1).reshape(B, (H//p)*(W//p), p*p*C)
        return x

    def forward(self, imgs):
        B = imgs.size(0)
        N = (self.cfg.img_size // self.cfg.patch_size) ** 2
        tokens = self.vit(imgs)  # (B,N,E)
        pred = self.pred_head(tokens)  # (B,N,patch_pixels)
        target = self.patchify(imgs)
        mask = self.random_mask(B, N, imgs.device)
        Bidx = torch.arange(B, device=imgs.device).unsqueeze(-1)
        loss = F.mse_loss(pred[Bidx, mask], target[Bidx, mask])
        return loss, {"mse_masked": loss.item(), "n_mask": mask.size(1)}
