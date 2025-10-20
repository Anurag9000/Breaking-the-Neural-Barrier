import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# Swin Transformer (minimal, canonical components)
# - Window MSA + Shifted Window MSA
# - Patch Merging for hierarchy
# ------------------------------

from typing import Tuple

def window_partition(x, win):
    B,H,W,C = x.shape
    x = x.view(B, H//win, win, W//win, win, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, win, win, C)
    return windows

def window_reverse(windows, win, H, W):
    B = windows.shape[0] // (H//win * W//win)
    x = windows.view(B, H//win, W//win, win, win, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0, drop=0.0):
        super().__init__()
        hid=int(dim*ratio)
        self.fc1=nn.Linear(dim,hid); self.act=nn.GELU(); self.fc2=nn.Linear(hid,dim); self.drop=nn.Dropout(drop)
    def forward(self,x):
        x=self.fc1(x); x=self.act(x); x=self.drop(x); x=self.fc2(x); x=self.drop(x); return x

class WindowAttention(nn.Module):
    def __init__(self, dim, heads, win, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.heads=heads; self.scale=(dim//heads)**-0.5; self.win=win
        self.qkv=nn.Linear(dim, dim*3, bias=True)
        self.proj=nn.Linear(dim, dim)
        self.ad=nn.Dropout(attn_drop); self.pd=nn.Dropout(proj_drop)

    def forward(self, x):  # x: (B*nW, win*win, C)
        Bn,N,C=x.shape
        qkv=self.qkv(x).reshape(Bn,N,3,self.heads,C//self.heads).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        attn=(q@k.transpose(-2,-1))*self.scale
        attn=attn.softmax(dim=-1); attn=self.ad(attn)
        x=(attn@v).transpose(1,2).reshape(Bn,N,C)
        x=self.proj(x); x=self.pd(x)
        return x

class SwinBlock(nn.Module):
    def __init__(self, dim, heads, win=7, shift=False):
        super().__init__()
        self.dim=dim; self.win=win; self.shift=shift
        self.n1=nn.LayerNorm(dim); self.attn=WindowAttention(dim, heads, win)
        self.n2=nn.LayerNorm(dim); self.mlp=MLP(dim)

    def forward(self, x, H, W):  # x: (B, H*W, C)
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        if self.shift:
            s=self.win//2
            x=torch.roll(x, shifts=(-s,-s), dims=(1,2))
        # partition
        windows=window_partition(x, self.win)             # (Bn, win, win, C)
        windows=windows.view(-1, self.win*self.win, C)
        # W-MSA
        xw=self.attn(self.n1(windows))
        xw=windows+xw
        # reverse
        xw=xw.view(-1,self.win,self.win,C)
        x=window_reverse(xw, self.win, H, W)
        if self.shift:
            s=self.win//2
            x=torch.roll(x, shifts=(s,s), dims=(1,2))
        x=x.view(B, H*W, C)
        # MLP
        x=x+self.mlp(self.n2(x))
        return x

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=96, patch=4):
        super().__init__()
        self.proj=nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm=nn.LayerNorm(embed_dim)
    def forward(self,x):
        x=self.proj(x)  # (B,C,H/4,W/4)
        B,C,H,W=x.shape
        x=x.flatten(2).transpose(1,2)
        x=self.norm(x)
        return x, H, W

class PatchMerge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction=nn.Linear(4*dim, 2*dim, bias=False)
        self.norm=nn.LayerNorm(4*dim)
    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.view(B, H, W, C)
        x = torch.cat([x[:,0::2,0::2,:], x[:,1::2,0::2,:], x[:,0::2,1::2,:], x[:,1::2,1::2,:]], dim=-1)
        x = x.view(B, -1, 4*C)
        x = self.norm(x)
        x = self.reduction(x)
        H, W = H//2, W//2
        return x, H, W

class Swin(nn.Module):
    def __init__(self, img_size=224, num_classes=10, embed_dim=96, depths=(2,2,6,2), heads=(3,6,12,24), win=7):
        super().__init__()
        self.patch = PatchEmbed(3, embed_dim, patch=4)
        self.stages=nn.ModuleList()
        dims=[embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        in_dim=embed_dim
        for i, d in enumerate(depths):
            blocks=[]
            for j in range(d):
                blocks += [SwinBlock(in_dim, heads[i], win=win, shift=(j%2==1))]
            self.stages.append(nn.ModuleList(blocks))
            if i < len(depths)-1:
                self.stages.append(PatchMerge(in_dim))
                in_dim*=2
        self.norm=nn.LayerNorm(in_dim)
        self.head=nn.Linear(in_dim, num_classes)

    def forward(self,x):
        x,H,W = self.patch(x)
        idx=0
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for blk in m:
                    x = blk(x,H,W)
            else:  # PatchMerge
                x,H,W = m(x,H,W)
        x=self.norm(x)
        x=x.mean(1)
        return self.head(x)
