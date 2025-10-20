import torch
import torch.nn as nn

# --------------------------------------------------
# VOLO (simplified): Outlook Attention blocks + classifier head
# --------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, patch=4, stride=4): super().__init__(); self.p=nn.Conv2d(in_ch,dim,patch,stride); self.n=nn.LayerNorm(dim)
    def forward(self,x): x=self.p(x); B,C,H,W=x.shape; t=x.flatten(2).transpose(1,2); t=self.n(t); return t,H,W

class OutlookAttention(nn.Module):
    def __init__(self, dim, heads=4, k=3):
        super().__init__(); self.h=heads; self.k=k; self.d=dim//heads; self.scale=self.d**-0.5
        self.vproj=nn.Linear(dim,dim); self.proj=nn.Linear(dim,dim)
    def forward(self,x,H,W):
        B,N,C=x.shape
        X=x.view(B,H,W,C)
        # sliding window aggregation of V (keys/queries implicit via local weights)
        pad=(self.k//2,)*4
        V=self.vproj(x).view(B,H,W,C).permute(0,3,1,2)  # (B,C,H,W)
        V=nn.functional.unfold(V, kernel_size=self.k, padding=self.k//2).permute(0,2,1)  # (B,N,C*k*k)
        V=V.view(B,N,self.h,self.d,self.k*self.k).permute(0,2,1,4,3)
        # learnable window weights per head
        w = nn.Parameter(torch.zeros(self.h, self.k*self.k, self.d, device=x.device))
        nn.init.trunc_normal_(w, std=0.02)
        out = (V * w.unsqueeze(0).unsqueeze(2)).sum(-2)  # (B,h,N,d)
        out = out.permute(0,2,1,3).reshape(B,N,C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, d, r=4.0): super().__init__(); h=int(d*r); self.fc1=nn.Linear(d,h); self.a=nn.GELU(); self.fc2=nn.Linear(h,d)
    def forward(self,x): return self.fc2(self.a(self.fc1(x)))

class VOLOBlock(nn.Module):
    def __init__(self, d,h): super().__init__(); self.n1=nn.LayerNorm(d); self.o=OutlookAttention(d,h); self.n2=nn.LayerNorm(d); self.m=MLP(d)
    def forward(self,x,H,W): x=x+self.o(self.n1(x),H,W); x=x+self.m(self.n2(x)); return x

class Down(nn.Module):
    def __init__(self, di, do): super().__init__(); self.c=nn.Conv2d(di,do,3,2,1); self.n=nn.LayerNorm(do)
    def forward(self,x,H,W): x=x.transpose(1,2).view(x.size(0),-1,H,W); x=self.c(x); B,C,H,W=x.shape; x=x.flatten(2).transpose(1,2); x=self.n(x); return x,H,W

class VOLO(nn.Module):
    def __init__(self, num_classes=10, embed=(64,128,256,512), depths=(2,2,6,2), heads=(2,4,8,16)):
        super().__init__(); self.pe=PatchEmbed(3,embed[0],4,4); self.stages=nn.ModuleList()
        for i in range(4):
            blks=nn.ModuleList([VOLOBlock(embed[i], heads[i]) for _ in range(depths[i])]); self.stages.append(blks)
            if i<3: self.stages.append(Down(embed[i], embed[i+1]))
        self.norm=nn.LayerNorm(embed[-1]); self.head=nn.Linear(embed[-1], num_classes)
    def forward(self,x): x,H,W=self.pe(x)
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for b in m: x=b(x,H,W)
            else: x,H,W=m(x,H,W)
        x=self.norm(x); x=x.mean(1); return self.head(x)
