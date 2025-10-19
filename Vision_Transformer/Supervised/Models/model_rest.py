import torch
import torch.nn as nn

# --------------------------------------------------
# ResT (ResNet-style stem + hierarchical transformer)
# --------------------------------------------------

class Stem(nn.Module):
    def __init__(self, c=64):
        super().__init__()
        self.net=nn.Sequential(
            nn.Conv2d(3,c//2,3,2,1), nn.BatchNorm2d(c//2), nn.ReLU(inplace=True),
            nn.Conv2d(c//2,c,3,2,1), nn.BatchNorm2d(c), nn.ReLU(inplace=True)
        )
    def forward(self,x): return self.net(x)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim):
        super().__init__(); self.p=nn.Conv2d(in_ch, dim, 1); self.n=nn.LayerNorm(dim)
    def forward(self,x): x=self.p(x); B,C,H,W=x.shape; x=x.flatten(2).transpose(1,2); x=self.n(x); return x,H,W

class MLP(nn.Module):
    def __init__(self, d, r=4.0): super().__init__(); h=int(d*r); self.fc1=nn.Linear(d,h); self.act=nn.GELU(); self.fc2=nn.Linear(h,d)
    def forward(self,x): return self.fc2(self.act(self.fc1(x)))

class Attn(nn.Module):
    def __init__(self, d, h): super().__init__(); self.h=h; self.d=d//h; self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
    def forward(self,x): B,N,D=x.shape; qkv=self.qkv(x).reshape(B,N,3,self.h,self.d).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]; a=(q@k.transpose(-2,-1))/self.d**0.5; a=a.softmax(-1); o=(a@v).transpose(1,2).reshape(B,N,D); return self.proj(o)

class Block(nn.Module):
    def __init__(self, d,h): super().__init__(); self.n1=nn.LayerNorm(d); self.a=Attn(d,h); self.n2=nn.LayerNorm(d); self.m=MLP(d)
    def forward(self,x): x=x+self.a(self.n1(x)); x=x+self.m(self.n2(x)); return x

class Down(nn.Module):
    def __init__(self, di, do): super().__init__(); self.c=nn.Conv2d(di,do,3,2,1); self.n=nn.LayerNorm(do)
    def forward(self,x,H,W): x=x.transpose(1,2).view(x.size(0),-1,H,W); x=self.c(x); B,C,H,W=x.shape; x=x.flatten(2).transpose(1,2); x=self.n(x); return x,H,W

class ResT(nn.Module):
    def __init__(self, num_classes=10, dims=(64,128,256,512), depths=(1,2,6,2), heads=(2,4,8,16)):
        super().__init__()
        self.stem=Stem(dims[0])
        self.p1=PatchEmbed(dims[0],dims[0])
        self.stages=nn.ModuleList()
        for i in range(4):
            blks=nn.ModuleList([Block(dims[i], heads[i]) for _ in range(depths[i])]); self.stages.append(blks)
            if i<3: self.stages.append(Down(dims[i], dims[i+1]))
        self.norm=nn.LayerNorm(dims[-1]); self.head=nn.Linear(dims[-1], num_classes)
    def forward(self,x):
        x=self.stem(x); x,H,W=self.p1(x)
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for b in m: x=b(x)
            else: x,H,W=m(x,H,W)
        x=self.norm(x); x=x.mean(1); return self.head(x)
