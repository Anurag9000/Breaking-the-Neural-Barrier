import torch
import torch.nn as nn

# ------------------------------
# PVT v2 (simplified)
# - Linear SRA flavor (depthwise conv in FFN, slight tweaks)
# ------------------------------

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, patch=4, stride=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, dim, kernel_size=patch, stride=stride)
        self.norm = nn.LayerNorm(dim)
    def forward(self,x):
        x=self.proj(x)
        B,C,H,W=x.shape
        x=x.flatten(2).transpose(1,2)
        x=self.norm(x)
        return x,H,W

class DWFFN(nn.Module):
    def __init__(self, dim, ratio=3.0, drop=0.0):
        super().__init__()
        hid=int(dim*ratio)
        self.fc1=nn.Linear(dim,hid)
        self.dwconv=nn.Conv2d(hid,hid,3,1,1,groups=hid)
        self.act=nn.GELU()
        self.fc2=nn.Linear(hid,dim)
        self.drop=nn.Dropout(drop)
    def forward(self,x,H,W):
        x=self.fc1(x)
        B,N,C=x.shape
        x=x.transpose(1,2).view(B,C,H,W)
        x=self.dwconv(x)
        x=self.act(x)
        x=x.flatten(2).transpose(1,2)
        x=self.drop(x)
        x=self.fc2(x)
        x=self.drop(x)
        return x

class LinearSRA(nn.Module):
    def __init__(self, dim, heads, sr_ratio=1):
        super().__init__()
        self.heads=heads; self.scale=(dim//heads)**-0.5; self.sr_ratio=sr_ratio
        self.q=nn.Linear(dim,dim); self.kv=nn.Linear(dim,dim*2)
        if sr_ratio>1:
            self.pool=nn.AvgPool2d(kernel_size=sr_ratio, stride=sr_ratio)
            self.norm=nn.LayerNorm(dim)
        else:
            self.pool=None
        self.proj=nn.Linear(dim,dim)
    def forward(self,x,H,W):
        B,N,C=x.shape
        q=self.q(x).reshape(B,N,self.heads,C//self.heads).permute(0,2,1,3)
        if self.pool is not None:
            xp=x.transpose(1,2).view(B,C,H,W)
            xp=self.pool(xp).flatten(2).transpose(1,2)
            xp=self.norm(xp)
            kv=self.kv(xp)
        else:
            kv=self.kv(x)
        kv=kv.reshape(B,-1,2,self.heads,C//self.heads).permute(2,0,3,1,4)
        k,v=kv[0],kv[1]
        attn=(q@k.transpose(-2,-1))*self.scale
        attn=attn.softmax(dim=-1)
        out=(attn@v).transpose(1,2).reshape(B,N,C)
        out=self.proj(out)
        return out

class Block(nn.Module):
    def __init__(self, dim, heads, sr_ratio):
        super().__init__()
        self.n1=nn.LayerNorm(dim); self.attn=LinearSRA(dim, heads, sr_ratio)
        self.n2=nn.LayerNorm(dim); self.ffn=DWFFN(dim)
    def forward(self,x,H,W):
        x=x+self.attn(self.n1(x),H,W)
        x=x+self.ffn(self.n2(x),H,W)
        return x

class PVTv2(nn.Module):
    def __init__(self, num_classes=10, embed_dims=(64,128,320,512), depths=(2,2,2,2), heads=(1,2,5,8), sr=(8,4,2,1)):
        super().__init__()
        self.p1=PatchEmbed(3,embed_dims[0],4,4)
        self.p2=PatchEmbed(embed_dims[0],embed_dims[1],2,2)
        self.p3=PatchEmbed(embed_dims[1],embed_dims[2],2,2)
        self.p4=PatchEmbed(embed_dims[2],embed_dims[3],2,2)
        self.s1=nn.ModuleList([Block(embed_dims[0],heads[0],sr[0]) for _ in range(depths[0])])
        self.s2=nn.ModuleList([Block(embed_dims[1],heads[1],sr[1]) for _ in range(depths[1])])
        self.s3=nn.ModuleList([Block(embed_dims[2],heads[2],sr[2]) for _ in range(depths[2])])
        self.s4=nn.ModuleList([Block(embed_dims[3],heads[3],sr[3]) for _ in range(depths[3])])
        self.norm=nn.LayerNorm(embed_dims[3])
        self.head=nn.Linear(embed_dims[3], num_classes)
    def forward(self,x):
        x,H,W=self.p1(x);   
        for b in self.s1: x=b(x,H,W)
        x,H,W=self.p2(x.transpose(1,2).view(x.size(0),-1,H,W))
        for b in self.s2: x=b(x,H,W)
        x,H,W=self.p3(x.transpose(1,2).view(x.size(0),-1,H,W))
        for b in self.s3: x=b(x,H,W)
        x,H,W=self.p4(x.transpose(1,2).view(x.size(0),-1,H,W))
        for b in self.s4: x=b(x,H,W)
        x=self.norm(x); x=x.mean(1)
        return self.head(x)
