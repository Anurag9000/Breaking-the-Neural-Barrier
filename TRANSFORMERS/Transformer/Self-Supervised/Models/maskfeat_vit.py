"""
MaskFeat-ViT (single-model):
- Predict hand-crafted features (e.g., HOG) over masked patches with a simple predictor.
- Here we implement HOG-like fixed features using Sobel gradients + pooling to build a low-dim target per patch.
"""
from __future__ import annotations
from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- simple fixed HOG-style feature extractor ---
class HOGFixed(nn.Module):
    def __init__(self, patch_size=4, bins=8):
        super().__init__()
        # Sobel filters
        gx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32)
        gy = gx.t()
        self.register_buffer('gx', gx.view(1,1,3,3))
        self.register_buffer('gy', gy.view(1,1,3,3))
        self.patch_size=patch_size; self.bins=bins

    def forward(self, imgs):
        # imgs: (B,3,H,W) -> grayscale
        gray = imgs.mean(1, keepdim=True)
        Gx = F.conv2d(gray, self.gx, padding=1)
        Gy = F.conv2d(gray, self.gy, padding=1)
        mag = torch.sqrt(Gx**2 + Gy**2 + 1e-6)
        ang = torch.atan2(Gy, Gx)  # (-pi, pi)
        ang = (ang + torch.pi) / (2*torch.pi)  # [0,1)
        # binning per patch
        B,_,H,W = mag.shape
        p = self.patch_size
        gh, gw = H//p, W//p
        mag = mag[:, :, :gh*p, :gw*p]
        ang = ang[:, :, :gh*p, :gw*p]
        mag = mag.reshape(B, 1, gh, p, gw, p)
        ang = ang.reshape(B, 1, gh, p, gw, p)
        # histogram per patch
        bins = []
        for b in range(self.bins):
            low = b/self.bins; high=(b+1)/self.bins
            w = (ang>=low) & (ang<high)
            bins.append((mag * w).sum(dim=(3,5)))  # sum within patch
        feat = torch.stack(bins, dim=-1)  # (B,1,gh,gw,bins)
        feat = feat.reshape(B, gh*gw, self.bins)
        # L2 normalize per patch
        feat = F.normalize(feat, dim=-1)
        return feat  # (B, N, bins)

# --- ViT encoder ---
class PatchEmbed(nn.Module):
    def __init__(self, img=32, patch=4, in_ch=3, dim=384):
        super().__init__(); self.num=(img//patch)**2
        self.proj=nn.Conv2d(in_ch,dim,kernel_size=patch,stride=patch)
    def forward(self,x): return self.proj(x).flatten(2).transpose(1,2)

class Block(nn.Module):
    def __init__(self, dim, heads=6, ratio=4.0):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=nn.MultiheadAttention(dim,heads,batch_first=True)
        self.n2=nn.LayerNorm(dim); self.mlp=nn.Sequential(nn.Linear(dim,int(dim*ratio)), nn.GELU(), nn.Linear(int(dim*ratio),dim))
    def forward(self,x): x=x+self.attn(self.n1(x),self.n1(x),self.n1(x))[0]; x=x+self.mlp(self.n2(x)); return x

class ViT(nn.Module):
    def __init__(self, img=32, patch=4, dim=384, depth=6, heads=6, ratio=4.0):
        super().__init__(); self.patch=PatchEmbed(img,patch,3,dim)
        self.pos=nn.Parameter(torch.zeros(1,self.patch.num,dim))
        self.blocks=nn.ModuleList([Block(dim,heads,ratio) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim); nn.init.trunc_normal_(self.pos,std=0.02)
    def forward(self,x):
        x=self.patch(x)+self.pos
        for b in self.blocks: x=b(x)
        return self.norm(x)  # (B,N,E)

@dataclass
class MaskFeatConfig:
    img:int=32; patch:int=4; dim:int=384; depth:int=6; heads:int=6; ratio:float=4.0
    bins:int=8; mask_ratio:float=0.6

class MaskFeatViT(nn.Module):
    def __init__(self, cfg: MaskFeatConfig):
        super().__init__(); self.cfg=cfg
        self.vit=ViT(cfg.img,cfg.patch,cfg.dim,cfg.depth,cfg.heads,cfg.ratio)
        self.hog=HOGFixed(cfg.patch, cfg.bins)
        self.pred=nn.Linear(cfg.dim, cfg.bins)

    def random_mask(self, B, N, device):
        n_mask=int(self.cfg.mask_ratio*N)
        idx=torch.rand(B,N,device=device).argsort(dim=1)
        return idx[:, :n_mask]

    def forward(self, imgs):
        B=imgs.size(0); N=(self.cfg.img//self.cfg.patch)**2
        tokens=self.vit(imgs)  # (B,N,E)
        pred=self.pred(tokens)  # (B,N,bins)
        with torch.no_grad():
            target=self.hog(imgs)  # (B,N,bins)
        mask=self.random_mask(B,N,imgs.device)
        Bidx=torch.arange(B,device=imgs.device).unsqueeze(-1)
        loss=F.mse_loss(pred[Bidx,mask], target[Bidx,mask])
        return loss, {"mse_masked": loss.item()}
