"""
VICReg with ViT (single-model)
- Invariance (MSE between view embeddings), Variance (std >= gamma), Covariance (off-diag penalty)
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, dim=384):
        super().__init__()
        self.num_patches=(img_size//patch_size)**2
        self.proj=nn.Conv2d(in_ch,dim,kernel_size=patch_size,stride=patch_size)
    def forward(self,x):
        return self.proj(x).flatten(2).transpose(1,2)

class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__(); h=int(dim*ratio)
        self.fc1=nn.Linear(dim,h); self.act=nn.GELU(); self.fc2=nn.Linear(h,dim)
    def forward(self,x): return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, heads=6, ratio=4.0):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.n2=nn.LayerNorm(dim); self.mlp=MLP(dim,ratio)
    def forward(self,x):
        x=x+self.attn(self.n1(x),self.n1(x),self.n1(x))[0]
        x=x+self.mlp(self.n2(x)); return x

class ViT(nn.Module):
    def __init__(self, img=32, patch=4, dim=384, depth=6, heads=6, ratio=4.0):
        super().__init__()
        self.patch=PatchEmbed(img,patch,3,dim)
        self.pos=nn.Parameter(torch.zeros(1,self.patch.num_patches,dim))
        self.blocks=nn.ModuleList([Block(dim,heads,ratio) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)
        nn.init.trunc_normal_(self.pos,std=0.02)
    def forward(self,x):
        x=self.patch(x)+self.pos
        for b in self.blocks: x=b(x)
        x=self.norm(x)
        return x.mean(1)

@dataclass
class VICRegConfig:
    img:int=32; patch:int=4; dim:int=384; depth:int=6; heads:int=6; ratio:float=4.0
    proj_hidden:int=2048; proj_out:int=2048
    w_inv:float=25.0; w_var:float=25.0; w_cov:float=1.0; gamma:float=1.0

class Projector(nn.Module):
    def __init__(self, in_dim, hidden, out_dim):
        super().__init__()
        self.net=nn.Sequential(
            nn.Linear(in_dim,hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden,hidden), nn.BatchNorm1d(hidden), nn.ReLU(inplace=True),
            nn.Linear(hidden,out_dim)
        )
    def forward(self,x): return self.net(x)

class VICRegViT(nn.Module):
    def __init__(self, cfg: VICRegConfig):
        super().__init__(); self.cfg=cfg
        self.enc=ViT(cfg.img,cfg.patch,cfg.dim,cfg.depth,cfg.heads,cfg.ratio)
        self.proj=Projector(cfg.dim,cfg.proj_hidden,cfg.proj_out)
    def forward(self, x1, x2):
        z1=self.proj(self.enc(x1)); z2=self.proj(self.enc(x2))
        inv = F.mse_loss(z1, z2)
        def std_loss(z):
            std = z.std(dim=0) + 1e-4
            return torch.mean(F.relu(self.cfg.gamma - std))
        def cov_loss(z):
            z = z - z.mean(dim=0)
            N = z.size(0)
            c = (z.T @ z) / (N - 1)
            off = c - torch.diag(torch.diag(c))
            return (off**2).sum() / z.size(1)
        var = std_loss(z1) + std_loss(z2)
        cov = cov_loss(z1) + cov_loss(z2)
        loss = self.cfg.w_inv*inv + self.cfg.w_var*var + self.cfg.w_cov*cov
        return loss, {"inv": inv.item(), "var": var.item(), "cov": cov.item()}
