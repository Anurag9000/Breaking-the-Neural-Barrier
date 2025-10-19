"""
SimCLR with ViT (single-model)
- InfoNCE contrastive loss with temperature tau
- Shared encoder + projector; two augmented views
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

class Block(nn.Module):
    def __init__(self, dim, heads=6, ratio=4.0):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.n2=nn.LayerNorm(dim)
        self.mlp=nn.Sequential(nn.Linear(dim,int(dim*ratio)), nn.GELU(), nn.Linear(int(dim*ratio),dim))
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
class SimCLRConfig:
    img:int=32; patch:int=4; dim:int=384; depth:int=6; heads:int=6; ratio:float=4.0
    proj_hidden:int=2048; proj_out:int=128; tau:float=0.2

class Projector(nn.Module):
    def __init__(self, in_dim, hid, out):
        super().__init__(); self.net=nn.Sequential(nn.Linear(in_dim,hid), nn.ReLU(True), nn.Linear(hid,out))
    def forward(self,x): return self.net(x)

class SimCLRVit(nn.Module):
    def __init__(self, cfg: SimCLRConfig):
        super().__init__(); self.cfg=cfg
        self.enc=ViT(cfg.img,cfg.patch,cfg.dim,cfg.depth,cfg.heads,cfg.ratio)
        self.proj=Projector(cfg.dim,cfg.proj_hidden,cfg.proj_out)
    def forward(self, x1, x2):
        z1=F.normalize(self.proj(self.enc(x1)), dim=1)
        z2=F.normalize(self.proj(self.enc(x2)), dim=1)
        z=torch.cat([z1,z2], dim=0)
        N=z.size(0)
        sim = z @ z.T  # (2B,2B)
        mask = torch.eye(N, device=z.device).bool()
        sim = sim[~mask].view(N, N-1)
        # positives: (i vs i+B) and (i+B vs i)
        B = z1.size(0)
        positives = torch.cat([torch.sum(z1*z2, dim=1), torch.sum(z2*z1, dim=1)], dim=0)
        logits = torch.cat([positives.unsqueeze(1), sim], dim=1)/self.cfg.tau
        labels = torch.zeros(N, dtype=torch.long, device=z.device)
        loss = F.cross_entropy(logits, labels)
        return loss, {"nce": loss.item()}
