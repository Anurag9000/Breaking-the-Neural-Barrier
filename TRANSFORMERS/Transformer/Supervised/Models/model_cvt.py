import torch
import torch.nn as nn

# ------------------------------
# CvT (Convolutional vision transformer)
# - Convolutional projections for Q/K/V
# - Hierarchical stages
# ------------------------------

class ConvProj(nn.Module):
    def __init__(self, in_dim, out_dim, kernel=3, stride=1, padding=1):
        super().__init__()
        self.proj = nn.Conv2d(in_dim, out_dim, kernel, stride, padding, bias=False)
        self.bn = nn.BatchNorm2d(out_dim)
        self.act = nn.GELU()
    def forward(self, x):
        return self.act(self.bn(self.proj(x)))

class ConvQKV(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads=heads; self.scale=(dim//heads)**-0.5
        self.q = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.k = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.v = nn.Conv2d(dim, dim, 1, 1, 0, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, 1, 0)
    def forward(self, x):  # x: (B, C, H, W)
        B,C,H,W=x.shape
        q=self.q(x).view(B,self.heads,C//self.heads,H*W)
        k=self.k(x).view(B,self.heads,C//self.heads,H*W)
        v=self.v(x).view(B,self.heads,C//self.heads,H*W)
        attn=(q.transpose(2,3)@k)/ (C//self.heads)**0.5
        attn=attn.softmax(dim=-1)
        out=attn@v.transpose(2,3)
        out=out.transpose(2,3).contiguous().view(B,C,H,W)
        return self.proj(out)

class Block(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.n1=nn.BatchNorm2d(dim); self.attn=ConvQKV(dim, heads)
        self.n2=nn.BatchNorm2d(dim); self.mlp=nn.Sequential(nn.Conv2d(dim, dim*4, 1), nn.GELU(), nn.Conv2d(dim*4, dim, 1))
    def forward(self,x):
        x=x+self.attn(self.n1(x))
        x=x+self.mlp(self.n2(x))
        return x

class CvT(nn.Module):
    def __init__(self, num_classes=10, dims=(64,192,384), heads=(1,3,6)):
        super().__init__()
        self.stem=ConvProj(3,dims[0],kernel=7,stride=4,padding=3)
        self.stage1=nn.Sequential(Block(dims[0],heads[0]), Block(dims[0],heads[0]))
        self.down1=ConvProj(dims[0],dims[1],stride=2)
        self.stage2=nn.Sequential(Block(dims[1],heads[1]), Block(dims[1],heads[1]))
        self.down2=ConvProj(dims[1],dims[2],stride=2)
        self.stage3=nn.Sequential(Block(dims[2],heads[2]), Block(dims[2],heads[2]))
        self.pool=nn.AdaptiveAvgPool2d(1)
        self.head=nn.Linear(dims[2], num_classes)
    def forward(self,x):
        x=self.stem(x)
        x=self.stage1(x)
        x=self.down1(x)
        x=self.stage2(x)
        x=self.down2(x)
        x=self.stage3(x)
        x=self.pool(x).flatten(1)
        return self.head(x)
