"""
Jigsaw-ViT (single-model)
- Split image into 3x3 tiles, apply a permutation from a fixed set, predict its index.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- ViT backbone ----
class PatchEmbed(nn.Module):
    def __init__(self,img=32,patch=4,in_ch=3,dim=384):
        super().__init__(); self.num=(img//patch)**2
        self.proj=nn.Conv2d(in_ch,dim,kernel_size=patch,stride=patch)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)

class Block(nn.Module):
    def __init__(self,dim,heads=6,ratio=4.0):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.n2=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,int(dim*ratio)), nn.GELU(), nn.Linear(int(dim*ratio),dim))
    def forward(self,x): x=x+self.attn(self.n1(x),self.n1(x),self.n1(x))[0]; x=x+self.mlp(self.n2(x)); return x

class ViT(nn.Module):
    def __init__(self,img=32,patch=4,dim=384,depth=6,heads=6,ratio=4.0):
        super().__init__(); self.patch=PatchEmbed(img,patch,3,dim)
        self.pos=nn.Parameter(torch.zeros(1,self.patch.num,dim))
        self.blocks=nn.ModuleList([Block(dim,heads,ratio) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim); nn.init.trunc_normal_(self.pos,std=0.02)
    def forward(self,x):
        x=self.patch(x)+self.pos
        for b in self.blocks: x=b(x)
        return self.norm(x).mean(1)

@dataclass
class JigsawConfig:
    img:int=32; patch:int=4; dim:int=384; depth:int=6; heads:int=6; ratio:float=4.0; num_perms:int=30

class JigsawViT(nn.Module):
    def __init__(self, cfg:JigsawConfig):
        super().__init__(); self.cfg=cfg
        self.enc=ViT(cfg.img,cfg.patch,cfg.dim,cfg.depth,cfg.heads,cfg.ratio)
        self.head=nn.Linear(cfg.dim, cfg.num_perms)
    def forward(self, x, y=None):
        z=self.enc(x)
        logits=self.head(z)
        if y is None: return logits, {}
        loss=F.cross_entropy(logits, y)
        return loss, {"ce": loss.item()}
