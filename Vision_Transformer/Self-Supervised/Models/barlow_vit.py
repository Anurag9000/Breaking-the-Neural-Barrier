"""
Barlow Twins with ViT (single-model):
- Two augmented views through the SAME encoder (shared weights)
- Projection head -> cross-correlation matrix -> invariance + redundancy reduction
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# Minimal ViT
class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_chans=3, dim=384):
        super().__init__()
        self.num_patches = (img_size//patch_size)**2
        self.proj = nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1,2)

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0):
        super().__init__()
        hid=int(dim*mlp_ratio)
        self.fc1=nn.Linear(dim,hid); self.act=nn.GELU(); self.fc2=nn.Linear(hid,dim)
    def forward(self,x): return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.n1=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim, heads, batch_first=True)
        self.n2=nn.LayerNorm(dim)
        self.mlp=MLP(dim, mlp_ratio)
    def forward(self,x):
        x = x + self.attn(self.n1(x), self.n1(x), self.n1(x))[0]
        x = x + self.mlp(self.n2(x))
        return x

class ViT(nn.Module):
    def __init__(self, img_size=32, patch_size=4, dim=384, depth=6, heads=6, mlp_ratio=4.0):
        super().__init__()
        self.patch=PatchEmbed(img_size,patch_size,3,dim)
        self.pos=nn.Parameter(torch.zeros(1,self.patch.num_patches,dim))
        self.blocks=nn.ModuleList([Block(dim,heads,mlp_ratio) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos,std=0.02)
    def forward(self,x):
        x=self.patch(x)+self.pos
        for b in self.blocks: x=b(x)
        x=self.norm(x)
        # global average over tokens
        return x.mean(dim=1)

@dataclass
class BarlowConfig:
    img_size:int=32; patch_size:int=4; dim:int=384; depth:int=6; heads:int=6; mlp_ratio:float=4.0
    proj_dim:int=8192; proj_hidden:int=4096; lambd:float=0.0051

class ProjectionMLP(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim, hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden, out_dim), nn.BatchNorm1d(out_dim, affine=False)
        )
    def forward(self,x): return self.net(x)

class BarlowTwinsViT(nn.Module):
    def __init__(self, cfg: BarlowConfig):
        super().__init__()
        self.cfg=cfg
        self.encoder=ViT(cfg.img_size,cfg.patch_size,cfg.dim,cfg.depth,cfg.heads,cfg.mlp_ratio)
        self.projector=ProjectionMLP(cfg.dim, cfg.proj_hidden, cfg.proj_dim)

    def forward(self, x1, x2):
        z1 = self.projector(self.encoder(x1))
        z2 = self.projector(self.encoder(x2))
        # Normalize across batch
        z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
        z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
        N = z1.size(0)
        c = (z1.T @ z2) / N  # (D,D)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
        loss = on_diag + self.cfg.lambd * off_diag
        return loss, {"on_diag": on_diag.item(), "off_diag": off_diag.item()}
