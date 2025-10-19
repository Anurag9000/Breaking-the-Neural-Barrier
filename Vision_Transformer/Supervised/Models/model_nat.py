import torch
import torch.nn as nn

# --------------------------------------------------
# NAT (Neighborhood Attention Transformer) - simplified
#   - Local sliding-window self-attention (neighborhood size k)
#   - Hierarchical stages with strided downsampling
# --------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, dim=64, patch=4, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, patch, stride)
        self.norm = nn.LayerNorm(dim)
    def forward(self,x):
        x=self.proj(x); B,C,H,W=x.shape
        t=x.flatten(2).transpose(1,2); t=self.norm(t)
        return t,H,W

class NATT(nn.Module):
    def __init__(self, dim, heads=4, k=7):
        super().__init__(); self.h=heads; self.k=k; self.d=dim//heads; self.scale=self.d**-0.5
        self.qkv=nn.Linear(dim, dim*3); self.proj=nn.Linear(dim, dim)
    def unfold(self, x, H, W):
        # x: (B,N,C) -> (B,C,H,W) -> neighborhoods
        B,N,C=x.shape
        x=x.transpose(1,2).view(B,C,H,W)
        pad=(self.k//2,)*4
        x=nn.functional.pad(x, (pad[0],pad[1],pad[2],pad[3]))
        unfold=nn.Unfold(kernel_size=self.k, stride=1)
        neigh=unfold(x)  # (B, C*k*k, H*W)
        neigh=neigh.transpose(1,2)  # (B, H*W, C*k*k)
        return neigh.view(B, H*W, C, self.k*self.k)
    def forward(self,x,H,W):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.h,self.d).permute(2,0,3,1,4)
        q,kbase,vbase=qkv[0],qkv[1],qkv[2]  # (B,h,N,d)
        neigh=self.unfold(x,H,W)  # (B,N,C,kk)
        # take k,v neighborhoods by projecting base to per-neigh tokens
        k = kbase.unsqueeze(-2).expand(B,self.h,N,self.k*self.k,self.d)
        v = vbase.unsqueeze(-2).expand(B,self.h,N,self.k*self.k,self.d)
        attn=(q.unsqueeze(-2)@k.transpose(-2,-1))*self.scale
        attn=attn.softmax(-1)
        out=(attn@v).squeeze(-2)  # (B,h,N,d)
        out=out.transpose(1,2).reshape(B,N,C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, dim, r=4.0): super().__init__(); h=int(dim*r); self.fc1=nn.Linear(dim,h); self.act=nn.GELU(); self.fc2=nn.Linear(h,dim)
    def forward(self,x): return self.fc2(self.act(self.fc1(x)))

class NATBlock(nn.Module):
    def __init__(self, dim, heads, k):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.attn=NATT(dim, heads, k); self.n2=nn.LayerNorm(dim); self.mlp=MLP(dim)
    def forward(self,x,H,W): x=x+self.attn(self.n1(x),H,W); x=x+self.mlp(self.n2(x)); return x

class Down(nn.Module):
    def __init__(self, dim_in, dim_out): super().__init__(); self.c=nn.Conv2d(dim_in,dim_out,3,2,1); self.n=nn.LayerNorm(dim_out)
    def forward(self,x,H,W): x=x.transpose(1,2).view(x.size(0),-1,H,W); x=self.c(x); B,C,H,W=x.shape; x=x.flatten(2).transpose(1,2); x=self.n(x); return x,H,W

class NAT(nn.Module):
    def __init__(self, num_classes=10, embed=(64,128,256,512), depths=(2,2,6,2), heads=(2,4,8,16), k=7):
        super().__init__(); self.pe=PatchEmbed(3,embed[0],4,4); self.stages=nn.ModuleList()
        for i in range(4):
            blks=nn.ModuleList([NATBlock(embed[i], heads[i], k) for _ in range(depths[i])]); self.stages.append(blks)
            if i<3: self.stages.append(Down(embed[i], embed[i+1]))
        self.norm=nn.LayerNorm(embed[-1]); self.head=nn.Linear(embed[-1], num_classes)
    def forward(self,x):
        x,H,W=self.pe(x)
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for b in m: x=b(x,H,W)
            else: x,H,W=m(x,H,W)
        x=self.norm(x); x=x.mean(1); return self.head(x)
