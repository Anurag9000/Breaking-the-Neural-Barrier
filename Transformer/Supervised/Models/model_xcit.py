import torch
import torch.nn as nn

# --------------------------------------------------
# XCiT (simplified): Cross-Covariance Attention (XCA)
#   - Attention over channels rather than tokens
# --------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, patch=16, stride=16): super().__init__(); self.p=nn.Conv2d(in_ch,dim,patch,stride); self.n=nn.LayerNorm(dim)
    def forward(self,x): x=self.p(x); B,C,H,W=x.shape; t=x.flatten(2).transpose(1,2); t=self.n(t); return t

class XCA(nn.Module):
    def __init__(self, dim, heads=8):
        super().__init__(); self.h=heads; self.d=dim//heads; self.scale=self.d**-0.5
        self.qkv=nn.Linear(dim, dim*3); self.proj=nn.Linear(dim, dim)
    def forward(self,x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.h,self.d).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]  # (B,h,N,d)
        # normalize across tokens -> channel correlation
        q=(q - q.mean(dim=-1, keepdim=True))
        k=(k - k.mean(dim=-1, keepdim=True))
        attn=(q.transpose(-2,-1)@k)/(N-1)
        attn=attn.softmax(-1)
        out=(attn@v.transpose(-2,-1)).transpose(-2,-1)  # back to (B,h,N,d)
        out=out.transpose(1,2).reshape(B,N,C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, d, r=4.0): super().__init__(); h=int(d*r); self.fc1=nn.Linear(d,h); self.a=nn.GELU(); self.fc2=nn.Linear(h,d)
    def forward(self,x): return self.fc2(self.a(self.fc1(x)))

class XBlock(nn.Module):
    def __init__(self,d,h): super().__init__(); self.n1=nn.LayerNorm(d); self.xca=XCA(d,h); self.n2=nn.LayerNorm(d); self.m=MLP(d)
    def forward(self,x): x=x+self.xca(self.n1(x)); x=x+self.m(self.n2(x)); return x

class XCiT(nn.Module):
    def __init__(self, num_classes=10, dim=384, depth=12, heads=8):
        super().__init__(); self.patch=PatchEmbed(3,dim,16,16); self.blocks=nn.ModuleList([XBlock(dim,heads) for _ in range(depth)]); self.n=nn.LayerNorm(dim); self.head=nn.Linear(dim,num_classes)
    def forward(self,x): x=self.patch(x); 
        for b in self.blocks: x=b(x)
        x=self.n(x).mean(1); return self.head(x)
