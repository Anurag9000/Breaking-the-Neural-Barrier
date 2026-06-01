import torch
import torch.nn as nn

# --------------------------------------------------
# Focal Transformer (simplified): multi-granularity attention
# --------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, patch=4, stride=4): super().__init__(); self.p=nn.Conv2d(in_ch,dim,patch,stride); self.n=nn.LayerNorm(dim)
    def forward(self,x): x=self.p(x); B,C,H,W=x.shape; t=x.flatten(2).transpose(1,2); t=self.n(t); return t,H,W

class MLP(nn.Module):
    def __init__(self, d, r=4.0): super().__init__(); h=int(d*r); self.fc1=nn.Linear(d,h); self.a=nn.GELU(); self.fc2=nn.Linear(h,d)
    def forward(self,x): return self.fc2(self.a(self.fc1(x)))

class FocalAttn(nn.Module):
    def __init__(self, dim, heads=4):
        super().__init__(); self.h=heads; self.d=dim//heads; self.scale=self.d**-0.5
        self.q=nn.Linear(dim,dim); self.kv=nn.Linear(dim,dim*2); self.proj=nn.Linear(dim,dim)
    def forward(self,x,H,W):
        B,N,C=x.shape
        q=self.q(x).reshape(B,N,self.h,self.d).permute(0,2,1,3)
        kv=self.kv(x).reshape(B,N,2,self.h,self.d).permute(2,0,3,1,4)
        k,v=kv[0],kv[1]
        # local window
        S=int((N)**0.5)
        win=4
        def window_pool(t):
            T=t.permute(0,1,3,2).reshape(B,self.h,self.d,S,S)
            T=nn.functional.avg_pool2d(T,(win,win),stride=win)
            return T.reshape(B,self.h,self.d,-1).permute(0,1,3,2)
        k_local=window_pool(k); v_local=window_pool(v)
        # global pooled
        k_global=k.mean(dim=2,keepdim=True); v_global=v.mean(dim=2,keepdim=True)
        k_all=torch.cat([k, k_local, k_global], dim=2)
        v_all=torch.cat([v, v_local, v_global], dim=2)
        attn=(q@k_all.transpose(-2,-1))*self.scale; attn=attn.softmax(-1)
        out=(attn@v_all).transpose(1,2).reshape(B,N,C)
        return self.proj(out)

class FocalBlock(nn.Module):
    def __init__(self, d, h): super().__init__(); self.n1=nn.LayerNorm(d); self.a=FocalAttn(d,h); self.n2=nn.LayerNorm(d); self.m=MLP(d)
    def forward(self,x,H,W): x=x+self.a(self.n1(x),H,W); x=x+self.m(self.n2(x)); return x

class Down(nn.Module):
    def __init__(self, di, do): super().__init__(); self.c=nn.Conv2d(di,do,3,2,1); self.n=nn.LayerNorm(do)
    def forward(self,x,H,W): x=x.transpose(1,2).view(x.size(0),-1,H,W); x=self.c(x); B,C,H,W=x.shape; x=x.flatten(2).transpose(1,2); x=self.n(x); return x,H,W

class FocalTransformer(nn.Module):
    def __init__(self, num_classes=10, embed=(64,128,256,512), depths=(2,2,6,2), heads=(2,4,8,16)):
        super().__init__(); self.pe=PatchEmbed(3,embed[0],4,4); self.stages=nn.ModuleList()
        for i in range(4):
            blks=nn.ModuleList([FocalBlock(embed[i], heads[i]) for _ in range(depths[i])]); self.stages.append(blks)
            if i<3: self.stages.append(Down(embed[i], embed[i+1]))
        self.norm=nn.LayerNorm(embed[-1]); self.head=nn.Linear(embed[-1], num_classes)
    def forward(self,x):
        x,H,W=self.pe(x)
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for b in m: x=b(x,H,W)
            else: x,H,W=m(x,H,W)
        x=self.norm(x); x=x.mean(1); return self.head(x)
