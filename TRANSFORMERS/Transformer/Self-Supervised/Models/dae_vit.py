"""
Denoising ViT Autoencoder (single-model)
- Add noise/cutout to inputs; reconstruct clean image patches
- Loss: MSE
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

class ViTEnc(nn.Module):
    def __init__(self,img=32,patch=4,dim=384,depth=6,heads=6,ratio=4.0):
        super().__init__(); self.patch=PatchEmbed(img,patch,3,dim)
        self.pos=nn.Parameter(torch.zeros(1,self.patch.num,dim))
        self.blocks=nn.ModuleList([Block(dim,heads,ratio) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim); nn.init.trunc_normal_(self.pos,std=0.02)
    def forward(self,x):
        x=self.patch(x)+self.pos
        for b in self.blocks: x=b(x)
        return self.norm(x)

@dataclass
class DAEConfig:
    img:int=32; patch:int=4; dim:int=384; depth:int=6; heads:int=6; ratio:float=4.0; noise_std:float=0.2; cutout_p:float=0.5

class ViTDAE(nn.Module):
    def __init__(self,cfg:DAEConfig):
        super().__init__(); self.cfg=cfg
        self.enc=ViTEnc(cfg.img,cfg.patch,cfg.dim,cfg.depth,cfg.heads,cfg.ratio)
        self.head=nn.Linear(cfg.dim,(cfg.patch**2)*3)
    def corrupt(self, imgs):
        x = imgs + self.cfg.noise_std*torch.randn_like(imgs)
        if self.cfg.cutout_p>0:
            B,C,H,W=x.shape
            m=torch.ones(B,1,H,W, device=x.device)
            cut = torch.rand(B, device=x.device) < self.cfg.cutout_p
            if cut.any():
                size = H//4
                cx = torch.randint(size, H-size, (cut.sum(),), device=x.device)
                cy = torch.randint(size, W-size, (cut.sum(),), device=x.device)
                x[cut, :, cx-size:cx+size, cy-size:cy+size] = 0
        return x.clamp(0,1)
    def patchify(self, imgs):
        p=self.cfg.patch; B,C,H,W=imgs.shape
        return imgs.reshape(B,C,H//p,p,W//p,p).permute(0,2,4,3,5,1).reshape(B,(H//p)*(W//p),p*p*C)
    def forward(self, imgs):
        noisy=self.corrupt(imgs)
        tokens=self.enc(noisy)
        pred=self.head(tokens)
        target=self.patchify(imgs)
        loss=F.mse_loss(pred,target)
        return loss,{"mse":loss.item()}
