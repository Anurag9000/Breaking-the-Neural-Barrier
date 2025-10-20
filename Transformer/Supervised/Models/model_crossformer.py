import torch
import torch.nn as nn

# --------------------------------------------------
# CrossFormer (simplified): cross-scale attention
#   - Within-scale MHSA + cross-scale token mixing via strided pooling
# --------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, patch=4, stride=4): super().__init__(); self.p=nn.Conv2d(in_ch,dim,patch,stride); self.n=nn.LayerNorm(dim)
    def forward(self,x): x=self.p(x); B,C,H,W=x.shape; t=x.flatten(2).transpose(1,2); t=self.n(t); return t,H,W

class MLP(nn.Module):
    def __init__(self, d, r=4.0): super().__init__(); h=int(d*r); self.fc1=nn.Linear(d,h); self.a=nn.GELU(); self.fc2=nn.Linear(h,d)
    def forward(self,x): return self.fc2(self.a(self.fc1(x)))

class Attn(nn.Module):
    def __init__(self, d, h): super().__init__(); self.h=h; self.ds=d//h; self.qkv=nn.Linear(d,3*d); self.proj=nn.Linear(d,d)
    def forward(self,x): B,N,D=x.shape; qkv=self.qkv(x).reshape(B,N,3,self.h,self.ds).permute(2,0,3,1,4); q,k,v=qkv[0],qkv[1],qkv[2]; a=(q@k.transpose(-2,-1))/self.ds**0.5; a=a.softmax(-1); o=(a@v).transpose(1,2).reshape(B,N,D); return self.proj(o)

class CrossScale(nn.Module):
    def __init__(self, d): super().__init__(); self.reduction=nn.Linear(4*d, d)
    def forward(self,x,H,W):
        B,N,D=x.shape; X=x.view(B,H,W,D)
        X=torch.cat([X[:,0::2,0::2,:], X[:,0::2,1::2,:], X[:,1::2,0::2,:], X[:,1::2,1::2,:]], dim=-1)
        X=X.view(B,-1,4*D)
        return self.reduction(X), H//2, W//2

class Block(nn.Module):
    def __init__(self, d,h): super().__init__(); self.n1=nn.LayerNorm(d); self.a=Attn(d,h); self.n2=nn.LayerNorm(d); self.m=MLP(d)
    def forward(self,x): x=x+self.a(self.n1(x)); x=x+self.m(self.n2(x)); return x

class CrossFormer(nn.Module):
    def __init__(self, num_classes=10, embed=(64,128,256,512), depths=(2,2,6,2), heads=(2,4,8,16)):
        super().__init__(); self.pe=PatchEmbed(3,embed[0],4,4); self.stages=nn.ModuleList()
        for i in range(4):
            blks=nn.ModuleList([Block(embed[i], heads[i]) for _ in range(depths[i])]); self.stages.append(blks)
            if i<3: self.stages.append(CrossScale(embed[i]))
        self.norm=nn.LayerNorm(embed[-1]); self.head=nn.Linear(embed[-1], num_classes)
    def forward(self,x): x,H,W=self.pe(x)
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for b in m: x=b(x)
            else: x,H,W=m(x,H,W)
        x=self.norm(x); x=x.mean(1); return self.head(x)
