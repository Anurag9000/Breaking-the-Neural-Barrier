import torch
import torch.nn as nn

# ------------------------------
# PVT v1 (Pyramid Vision Transformer)
# - Patch embeddings with stride (hierarchical)
# - Spatial-Reduction Attention (SRA): reduce K,V via strided conv
# ------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, patch=4, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=stride, padding=0)
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        x = self.proj(x)
        B,C,H,W = x.shape
        x = x.flatten(2).transpose(1,2)
        x = self.norm(x)
        return x, H, W

class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0, drop=0.0):
        super().__init__()
        hid=int(dim*ratio)
        self.fc1=nn.Linear(dim,hid); self.act=nn.GELU(); self.fc2=nn.Linear(hid,dim); self.drop=nn.Dropout(drop)
    def forward(self,x):
        x=self.fc1(x); x=self.act(x); x=self.drop(x); x=self.fc2(x); x=self.drop(x); return x

class SRAttn(nn.Module):
    def __init__(self, dim, heads, sr_ratio=1, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.heads=heads; self.scale=(dim//heads)**-0.5; self.sr_ratio=sr_ratio
        self.q = nn.Linear(dim, dim)
        self.kv = nn.Linear(dim, dim*2)
        if sr_ratio>1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        else:
            self.sr = None
        self.proj = nn.Linear(dim, dim)
        self.ad=nn.Dropout(attn_drop); self.pd=nn.Dropout(proj_drop)

    def forward(self, x, H, W):  # x: (B,N,C)
        B,N,C=x.shape
        q=self.q(x).reshape(B,N,self.heads,C//self.heads).permute(0,2,1,3)
        if self.sr is not None:
            xp=x.transpose(1,2).view(B,C,H,W)
            xp=self.sr(xp).flatten(2).transpose(1,2)
            xp=self.norm(xp)
            kv=self.kv(xp)
        else:
            kv=self.kv(x)
        kv=kv.reshape(B,-1,2,self.heads,C//self.heads).permute(2,0,3,1,4)
        k,v=kv[0],kv[1]
        attn=(q@k.transpose(-2,-1))*self.scale
        attn=attn.softmax(dim=-1); attn=self.ad(attn)
        out=(attn@v).transpose(1,2).reshape(B,N,C)
        out=self.proj(out); out=self.pd(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=SRAttn(dim, heads, sr_ratio)
        self.n2=nn.LayerNorm(dim); self.mlp=MLP(dim)
    def forward(self,x,H,W):
        x=x+self.attn(self.n1(x),H,W)
        x=x+self.mlp(self.n2(x))
        return x

class PVTv1(nn.Module):
    def __init__(self, img_size=224, num_classes=10, embed_dims=(64,128,320,512), depths=(2,2,2,2), heads=(1,2,5,8), sr_ratios=(8,4,2,1)):
        super().__init__()
        self.patch1=PatchEmbed(3, embed_dims[0], patch=4, stride=4)
        self.patch2=PatchEmbed(embed_dims[0], embed_dims[1], patch=2, stride=2)
        self.patch3=PatchEmbed(embed_dims[1], embed_dims[2], patch=2, stride=2)
        self.patch4=PatchEmbed(embed_dims[2], embed_dims[3], patch=2, stride=2)
        self.stage1=nn.ModuleList([Block(embed_dims[0], heads[0], sr_ratios[0]) for _ in range(depths[0])])
        self.stage2=nn.ModuleList([Block(embed_dims[1], heads[1], sr_ratios[1]) for _ in range(depths[1])])
        self.stage3=nn.ModuleList([Block(embed_dims[2], heads[2], sr_ratios[2]) for _ in range(depths[2])])
        self.stage4=nn.ModuleList([Block(embed_dims[3], heads[3], sr_ratios[3]) for _ in range(depths[3])])
        self.norm=nn.LayerNorm(embed_dims[3])
        self.head=nn.Linear(embed_dims[3], num_classes)

    def forward(self,x):
        x,H,W=self.patch1(x)
        for b in self.stage1: x=b(x,H,W)
        x,H,W=self.patch2(x.transpose(1,2).view(x.size(0),-1,H,W))
        for b in self.stage2: x=b(x,H,W)
        x,H,W=self.patch3(x.transpose(1,2).view(x.size(0),-1,H,W))
        for b in self.stage3: x=b(x,H,W)
        x,H,W=self.patch4(x.transpose(1,2).view(x.size(0),-1,H,W))
        for b in self.stage4: x=b(x,H,W)
        x=self.norm(x); x=x.mean(1)
        return self.head(x)
