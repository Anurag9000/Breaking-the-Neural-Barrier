import torch
import torch.nn as nn

# --------------------------------------------------
# LeViT (simplified): conv stem + transformer blocks; lightweight
# --------------------------------------------------

class Stem(nn.Module):
    def __init__(self, c=64): super().__init__(); self.net=nn.Sequential(nn.Conv2d(3,c,3,2,1), nn.BatchNorm2d(c), nn.SiLU(), nn.Conv2d(c,c,3,1,1), nn.BatchNorm2d(c), nn.SiLU())
    def forward(self,x): return self.net(x)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, stride=2): super().__init__(); self.c=nn.Conv2d(in_ch,dim,3,stride,1); self.n=nn.LayerNorm(dim)
    def forward(self,x): x=self.c(x); B,C,H,W=x.shape; t=x.flatten(2).transpose(1,2); t=self.n(t); return t,H,W

class MLP(nn.Module):
    def __init__(self, d, h=2.0): super().__init__(); hid=int(d*h); self.fc1=nn.Linear(d,hid); self.a=nn.Hardswish(); self.fc2=nn.Linear(hid,d)
    def forward(self,x): return self.fc2(self.a(self.fc1(x)))

class Attn(nn.Module):
    def __init__(self, d, heads): super().__init__(); self.h=heads; self.ds=d//heads; self.q=nn.Linear(d,d); self.kv=nn.Linear(d,2*d); self.proj=nn.Linear(d,d)
    def forward(self,x): B,N,D=x.shape; q=self.q(x).reshape(B,N,self.h,self.ds).permute(0,2,1,3); kv=self.kv(x).reshape(B,N,2,self.h,self.ds).permute(2,0,3,1,4); k,v=kv[0],kv[1]; a=(q@k.transpose(-2,-1))/self.ds**0.5; a=a.softmax(-1); o=(a@v).transpose(1,2).reshape(B,N,D); return self.proj(o)

class Block(nn.Module):
    def __init__(self, d,h): super().__init__(); self.n1=nn.LayerNorm(d); self.a=Attn(d,h); self.n2=nn.LayerNorm(d); self.m=MLP(d)
    def forward(self,x): x=x+self.a(self.n1(x)); x=x+self.m(self.n2(x)); return x

class LeViT(nn.Module):
    def __init__(self, num_classes=10, dims=(128,256,384), depths=(2,3,4), heads=(4,6,8)):
        super().__init__(); self.stem=Stem(dims[0]//2); self.p1=PatchEmbed(dims[0]//2, dims[0], stride=2)
        stages=[]
        for i,d in enumerate(depths):
            stages += [Block(dims[i], heads[i]) for _ in range(d)]
        self.blocks=nn.Sequential(*stages)
        self.down2=PatchEmbed(dims[0], dims[1], stride=2)
        self.tail=[Block(dims[1], heads[1]) for _ in range(depths[1])]
        self.down3=PatchEmbed(dims[1], dims[2], stride=2)
        self.tail2=[Block(dims[2], heads[2]) for _ in range(depths[2])]
        self.norm=nn.LayerNorm(dims[-1]); self.head=nn.Linear(dims[-1], num_classes)
    def forward(self,x):
        x=self.stem(x); x,H,W=self.p1(x); x=self.blocks(x)
        x,H,W=self.down2(x.transpose(1,2).view(x.size(0),-1,H,W));
        for b in self.tail: x=b(x)
        x,H,W=self.down3(x.transpose(1,2).view(x.size(0),-1,H,W));
        for b in self.tail2: x=b(x)
        x=self.norm(x).mean(1); return self.head(x)
