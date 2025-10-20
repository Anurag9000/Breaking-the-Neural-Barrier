"""
Colorization-ViT (single-model)
- Input: grayscale L
- Target: ab chroma (Lab space), predicted per patch
- Loss: MSE in ab space
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---- RGB<->Lab utilities (approx) ----
# Source: standard formulas adapted to torch

def _rgb_to_xyz(rgb):
    # rgb in [0,1]
    mask = rgb > 0.04045
    rgb_lin = torch.where(mask, ((rgb + 0.055)/1.055)**2.4, rgb/12.92)
    r,g,b = rgb_lin[:,0], rgb_lin[:,1], rgb_lin[:,2]
    x = 0.4124*r + 0.3576*g + 0.1805*b
    y = 0.2126*r + 0.7152*g + 0.0722*b
    z = 0.0193*r + 0.1192*g + 0.9505*b
    return torch.stack([x,y,z], dim=1)

def _xyz_to_lab(xyz):
    # D65 white point
    xn, yn, zn = 0.95047, 1.0, 1.08883
    x, y, z = xyz[:,0]/xn, xyz[:,1]/yn, xyz[:,2]/zn
    eps = 216/24389; k = 24389/27
    def f(t):
        return torch.where(t>eps, t.pow(1/3), (k*t+16)/116)
    fx, fy, fz = f(x), f(y), f(z)
    L = 116*fy - 16
    a = 500*(fx - fy)
    b = 200*(fy - fz)
    return torch.stack([L,a,b], dim=1)

def rgb_to_lab(img):
    # img: (B,3,H,W) in [0,1]
    B,C,H,W = img.shape
    flat = img.permute(0,2,3,1).reshape(B*H*W, 3)
    lab = _xyz_to_lab(_rgb_to_xyz(flat))
    lab = lab.reshape(B,H,W,3).permute(0,3,1,2)
    return lab

# ---- ViT backbone ----
class PatchEmbed(nn.Module):
    def __init__(self,img=32,patch=4,in_ch=1,dim=384):
        super().__init__(); self.num=(img//patch)**2; self.p=patch
        self.proj=nn.Conv2d(in_ch,dim,kernel_size=patch,stride=patch)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)

class Block(nn.Module):
    def __init__(self,dim,heads=6,ratio=4.0):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.n2=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,int(dim*ratio)), nn.GELU(), nn.Linear(int(dim*ratio),dim))
    def forward(self,x): x=x+self.attn(self.n1(x),self.n1(x),self.n1(x))[0]; x=x+self.mlp(self.n2(x)); return x

class ViTEnc(nn.Module):
    def __init__(self,img=32,patch=4,dim=384,depth=6,heads=6,ratio=4.0):
        super().__init__(); self.patch=PatchEmbed(img,patch,1,dim)
        self.pos=nn.Parameter(torch.zeros(1,self.patch.num,dim))
        self.blocks=nn.ModuleList([Block(dim,heads,ratio) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim); nn.init.trunc_normal_(self.pos,std=0.02)
    def forward(self,x):
        x=self.patch(x)+self.pos
        for b in self.blocks: x=b(x)
        return self.norm(x)

@dataclass
class ColorConfig:
    img:int=32; patch:int=4; dim:int=384; depth:int=6; heads:int=6; ratio:float=4.0

class ColorizationViT(nn.Module):
    def __init__(self,cfg:ColorConfig):
        super().__init__(); self.cfg=cfg
        self.enc=ViTEnc(cfg.img,cfg.patch,cfg.dim,cfg.depth,cfg.heads,cfg.ratio)
        self.head=nn.Linear(cfg.dim, (cfg.patch**2)*2)  # predict ab per patch
    def patchify(self, x):
        p=self.cfg.patch; B,C,H,W=x.shape
        return x.reshape(B,C,H//p,p,W//p,p).permute(0,2,4,3,5,1).reshape(B,(H//p)*(W//p),p*p*C)
    def forward(self, imgs_rgb):
        imgs = imgs_rgb.clamp(0,1)
        with torch.no_grad():
            lab = rgb_to_lab(imgs)
            L = lab[:, :1] / 100.0  # scale L to [0,1]
            ab = lab[:, 1:] / 128.0  # roughly in [-1,1] -> [-1,1], keep as is for MSE stability
        tokens=self.enc(L)
        pred=self.head(tokens).reshape(imgs.size(0), -1, self.cfg.patch**2, 2)
        target=self.patchify(ab).reshape(imgs.size(0), -1, self.cfg.patch**2, 2)
        loss=F.mse_loss(pred, target)
        return loss, {"mse_ab": loss.item()}
