import torch
import torch.nn as nn

# --------------------------------------------------
# DAT (Deformable Attention for images) - very simplified
#  - Learn offsets for sampling key/value via grid_sample
#  - Single-scale hierarchical backbone for classification
# --------------------------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, dim=96, patch=4, stride=4):
        super().__init__()
        self.proj=nn.Conv2d(in_ch, dim, patch, stride)
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        x=self.proj(x); B,C,H,W=x.shape
        t=x.flatten(2).transpose(1,2); t=self.norm(t)
        return t,H,W

class DeformAttn(nn.Module):
    def __init__(self, dim, heads=4, n_points=4):
        super().__init__()
        self.h=heads; self.d=dim//heads; self.np=n_points
        self.q=nn.Linear(dim, dim); self.kv=nn.Linear(dim, dim*2)
        self.offset = nn.Linear(dim, heads*n_points*2)  # (dx, dy)
        self.proj=nn.Linear(dim, dim)
    def forward(self,x,H,W):
        B,N,C=x.shape
        q=self.q(x).reshape(B,N,self.h,self.d).permute(0,2,1,3)  # (B,h,N,d)
        kv=self.kv(x).reshape(B,N,2,self.h,self.d).permute(2,0,3,1,4)
        k,v=kv[0],kv[1]  # (B,h,N,d)
        # base reference grid (normalized)
        ys, xs = torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        base = torch.stack([xs, ys], dim=-1).view(1,H*W,2)  # (1,N,2)
        off = self.offset(x).view(B,N,self.h,self.np,2)
        grids = (base.unsqueeze(2).unsqueeze(3) + off).clamp(-1,1)  # (B,N,h,np,2)
        feat = x.transpose(1,2).view(B,C,H,W)
        # sample v features at offsets per head then aggregate by dot with q
        vmap = v.permute(0,2,1,3).reshape(B,N,C)  # (B,N,C)
        vfeat = feat  # (B,C,H,W)
        gathered = []
        for p in range(self.np):
            g = grids[..., p, :].view(B, N, self.h, 1, 2)
            g = g.view(B*self.h, N, 1, 2)
            samp = nn.functional.grid_sample(vfeat.repeat(self.h,1,1,1), g.view(B*self.h, H, W, 2).permute(0,2,1,3), align_corners=True, mode='bilinear', padding_mode='zeros')
            samp = samp.view(B,self.h,C,1,1).squeeze(-1).squeeze(-1).permute(0,1,2)
            gathered.append(samp)
        vg = torch.stack(gathered, dim=-1).mean(-1)  # (B,h,C)
        # attention
        attn = (q @ k.transpose(-2,-1)) * (self.d ** -0.5)
        attn = attn.softmax(-1)
        out = (attn @ vg)  # (B,h,N,C) with broadcast on C via vg
        out = out.permute(0,2,1,3).reshape(B,N,C)
        return self.proj(out)

class MLP(nn.Module):
    def __init__(self, dim, r=4.0):
        super().__init__(); h=int(dim*r); self.fc1=nn.Linear(dim,h); self.act=nn.GELU(); self.fc2=nn.Linear(h,dim)
    def forward(self,x): return self.fc2(self.act(self.fc1(x)))

class DATBlock(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.n1=nn.LayerNorm(dim); self.da=DeformAttn(dim, heads); self.n2=nn.LayerNorm(dim); self.mlp=MLP(dim)
    def forward(self,x,H,W): x=x+self.da(self.n1(x),H,W); x=x+self.mlp(self.n2(x)); return x

class Down(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__(); self.c=nn.Conv2d(dim_in, dim_out, 3, 2, 1); self.n=nn.LayerNorm(dim_out)
    def forward(self,x,H,W): x=x.transpose(1,2).view(x.size(0),-1,H,W); x=self.c(x); B,C,H,W=x.shape; x=x.flatten(2).transpose(1,2); x=self.n(x); return x,H,W

class DAT(nn.Module):
    def __init__(self, num_classes=10, embed=(96,192,384,512), depths=(2,2,6,2), heads=(3,6,12,16)):
        super().__init__()
        self.pe=PatchEmbed(3,embed[0],4,4)
        self.stages=nn.ModuleList()
        for i in range(4):
            blks=nn.ModuleList([DATBlock(embed[i], heads[i]) for _ in range(depths[i])])
            self.stages.append(blks)
            if i<3: self.stages.append(Down(embed[i], embed[i+1]))
        self.norm=nn.LayerNorm(embed[-1]); self.head=nn.Linear(embed[-1], num_classes)
    def forward(self,x):
        x,H,W=self.pe(x)
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for b in m: x=b(x,H,W)
            else:
                x,H,W=m(x,H,W)
        x=self.norm(x); x=x.mean(1)
        return self.head(x)
