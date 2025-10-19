"""
SimSiam with ViT (single-model)
- Shared encoder+projector; predictor on one branch; stop-grad on the other.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img=32, patch=4, in_ch=3, dim=384):
        super().__init__(); self.num=(img//patch)**2
        self.proj=nn.Conv2d(in_ch,dim,kernel_size=patch,stride=patch)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)

class MLPBlock(nn.Module):
    def __init__(self, dim, ratio=4.0):
        super().__init__(); h=int(dim*ratio)
        self.fc1=nn.Linear(dim,h); self.act=nn.GELU(); self.fc2=nn.Linear(h,dim)
    def forward(self,x): return self.fc2(self.act(self.fc1(x)))

class Block(nn.Module):
    def __init__(self, dim, heads=6, ratio=4.0):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.n2=nn.LayerNorm(dim); self.mlp=MLPBlock(dim,ratio)
    def forward(self,x):
        x=x+self.attn(self.n1(x),self.n1(x),self.n1(x))[0]
        x=x+self.mlp(self.n2(x)); return x

class ViT(nn.Module):
    def __init__(self, img=32, patch=4, dim=384, depth=6, heads=6, ratio=4.0):
        super().__init__(); self.patch=PatchEmbed(img,patch,3,dim)
        self.pos=nn.Parameter(torch.zeros(1,self.patch.num,dim))
        self.blocks=nn.ModuleList([Block(dim,heads,ratio) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim); nn.init.trunc_normal_(self.pos,std=0.02)
    def forward(self,x):
        x=self.patch(x)+self.pos
        for b in self.blocks: x=b(x)
        return self.norm(x).mean(1)

@dataclass
class SimSiamConfig:
    img:int=32; patch:int=4; dim:int=384; depth:int=6; heads:int=6; ratio:float=4.0
    proj_hidden:int=2048; proj_out:int=2048; pred_hidden:int=512

class Projector(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,hid), nn.BatchNorm1d(hid), nn.ReLU(True), nn.Linear(hid,out))
    def forward(self,x): return self.net(x)

class Predictor(nn.Module):
    def __init__(self, dim, hid):
        super().__init__(); self.net=nn.Sequential(nn.Linear(dim,hid), nn.BatchNorm1d(hid), nn.ReLU(True), nn.Linear(hid,dim))
    def forward(self,x): return self.net(x)

class SimSiamViT(nn.Module):
    def __init__(self, cfg: SimSiamConfig):
        super().__init__(); self.cfg=cfg
        self.enc=ViT(cfg.img,cfg.patch,cfg.dim,cfg.depth,cfg.heads,cfg.ratio)
        self.proj=Projector(cfg.dim,cfg.proj_hidden,cfg.proj_out)
        self.pred=Predictor(cfg.proj_out,cfg.pred_hidden)
    def forward(self, x1, x2):
        z1=self.proj(self.enc(x1)); z2=self.proj(self.enc(x2))
        p1=self.pred(z1); p2=self.pred(z2)
        # stop-grad on the target branch
        loss = -(F.cosine_similarity(p1, z2.detach(), dim=1).mean() + F.cosine_similarity(p2, z1.detach(), dim=1).mean()) * 0.5
        return loss, {"cos": -loss.item()}
