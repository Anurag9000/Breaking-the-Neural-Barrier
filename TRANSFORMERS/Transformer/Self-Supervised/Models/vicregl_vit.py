"""
VICRegL with ViT (single-model)
- Global invariance + local (token/patch) terms
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

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
    def forward(self,x, return_tokens=False):
        x=self.patch(x)+self.pos
        for b in self.blocks: x=b(x)
        x=self.norm(x)
        if return_tokens: return x
        return x.mean(1)

@dataclass
class VICRegLConfig:
    img:int=32; patch:int=4; dim:int=384; depth:int=6; heads:int=6; ratio:float=4.0
    proj_hidden:int=2048; proj_out:int=1024
    w_inv:float=25.0; w_var:float=25.0; w_cov:float=1.0; w_local:float=1.0; gamma:float=1.0

class Projector(nn.Module):
    def __init__(self,in_dim,hid,out):
        super().__init__(); self.net=nn.Sequential(
            nn.Linear(in_dim,hid), nn.BatchNorm1d(hid), nn.ReLU(True),
            nn.Linear(hid,hid), nn.BatchNorm1d(hid), nn.ReLU(True),
            nn.Linear(hid,out)
        )
    def forward(self,x): return self.net(x)

class VICRegLViT(nn.Module):
    def __init__(self,cfg:VICRegLConfig):
        super().__init__(); self.cfg=cfg
        self.enc=ViT(cfg.img,cfg.patch,cfg.dim,cfg.depth,cfg.heads,cfg.ratio)
        self.proj=Projector(cfg.dim,cfg.proj_hidden,cfg.proj_out)
        self.local_head=nn.Linear(cfg.dim, cfg.proj_out)

    def _var_cov(self, z):
        std = z.std(dim=0) + 1e-4
        var = torch.mean(F.relu(self.cfg.gamma - std))
        zc = z - z.mean(dim=0)
        c = (zc.T @ zc) / (z.size(0)-1)
        cov = (c - torch.diag(torch.diag(c))).pow(2).sum()/z.size(1)
        return var, cov

    def forward(self, x1, x2):
        # global
        g1 = self.proj(self.enc(x1))
        g2 = self.proj(self.enc(x2))
        inv = F.mse_loss(g1,g2)
        var1,cov1 = self._var_cov(g1); var2,cov2 = self._var_cov(g2)
        # local (token-wise)
        t1 = self.enc(x1, return_tokens=True)
        t2 = self.enc(x2, return_tokens=True)
        # mean over tokens of MSE between corresponding tokens (no teacher)
        l1 = self.local_head(t1); l2 = self.local_head(t2)
        local = F.mse_loss(l1, l2)
        loss = self.cfg.w_inv*inv + self.cfg.w_var*(var1+var2) + self.cfg.w_cov*(cov1+cov2) + self.cfg.w_local*local
        return loss, {"inv":inv.item(), "local": local.item()}
