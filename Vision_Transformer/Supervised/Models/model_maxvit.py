import torch
import torch.nn as nn

# --------------------------------------------------
# MaxViT (very simplified):
#  - MBConv -> Block Attention (window) -> Grid Attention
# --------------------------------------------------

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, s=1, exp=4):
        super().__init__()
        hid=in_ch*exp
        self.block=nn.Sequential(
            nn.Conv2d(in_ch,hid,1,1,0,bias=False), nn.BatchNorm2d(hid), nn.SiLU(),
            nn.Conv2d(hid,hid,3,s,1,groups=hid,bias=False), nn.BatchNorm2d(hid), nn.SiLU(),
            nn.Conv2d(hid,out_ch,1,1,0,bias=False), nn.BatchNorm2d(out_ch)
        )
        self.ds = s>1 or in_ch!=out_ch
        if self.ds: self.skip=nn.Sequential(nn.Conv2d(in_ch,out_ch,1,s,0,bias=False), nn.BatchNorm2d(out_ch))
    def forward(self,x):
        y=self.block(x); x=self.skip(x) if hasattr(self,'skip') else x; return torch.relu(x+y)

class WindowAttention2D(nn.Module):
    def __init__(self, dim, heads, win=7):
        super().__init__(); self.h=heads; self.win=win; self.scale=(dim//heads)**-0.5
        self.qkv=nn.Conv2d(dim, dim*3, 1); self.proj=nn.Conv2d(dim, dim, 1)
    def forward(self,x):
        B,C,H,W=x.shape
        padH=(self.win - H%self.win)%self.win; padW=(self.win - W%self.win)%self.win
        x=nn.functional.pad(x,(0,padW,0,padH))
        H2,W2=x.shape[-2:]
        x=x.view(B,C,H2//self.win,self.win,W2//self.win,self.win).permute(0,2,4,1,3,5).contiguous().view(-1,C,self.win,self.win)
        qkv=self.qkv(x).view(-1,3,self.h,C//self.h,self.win*self.win).permute(1,0,2,4,3)
        q,k,v=qkv[0],qkv[1],qkv[2]
        attn=(q@k.transpose(-2,-1))*self.scale; attn=attn.softmax(-1)
        out=(attn@v).reshape(-1,self.h,self.win,self.win,C//self.h).permute(0,1,4,2,3).contiguous().view(-1,C,self.win,self.win)
        out=self.proj(out)
        out=out.view(B,H2//self.win,W2//self.win,C,self.win,self.win).permute(0,3,1,4,2,5).contiguous().view(B,C,H2,W2)
        return out[..., :H, :W]

class GridAttention2D(nn.Module):
    def __init__(self, dim, heads, grid=7):
        super().__init__(); self.h=heads; self.grid=grid; self.scale=(dim//heads)**-0.5
        self.qkv=nn.Conv2d(dim, dim*3, 1); self.proj=nn.Conv2d(dim, dim, 1)
    def forward(self,x):
        B,C,H,W=x.shape
        gh=max(1,H//self.grid); gw=max(1,W//self.grid)
        xs=nn.functional.adaptive_avg_pool2d(x,(gh,gw))
        qkv=self.qkv(xs).view(B,3,self.h,C//self.h,gh*gw).permute(1,0,2,4,3)
        q,k,v=qkv[0],qkv[1],qkv[2]
        a=(q@k.transpose(-2,-1))*self.scale; a=a.softmax(-1)
        y=(a@v).permute(0,1,3,2).reshape(B,C,gh,gw)
        y=nn.functional.interpolate(y,size=(H,W),mode='bilinear',align_corners=False)
        y=self.proj(y)
        return y

class MaxViTBlock(nn.Module):
    def __init__(self, dim, heads, win=7, grid=7, s=1):
        super().__init__()
        self.mb=MBConv(dim,dim,s=s)
        self.n1=nn.BatchNorm2d(dim); self.wmsa=WindowAttention2D(dim,heads,win)
        self.n2=nn.BatchNorm2d(dim); self.gsa=GridAttention2D(dim,heads,grid)
    def forward(self,x):
        x=self.mb(x)
        x=x+self.wmsa(self.n1(x))
        x=x+self.gsa(self.n2(x))
        return x

class MaxViT(nn.Module):
    def __init__(self, num_classes=10, dims=(64,128,256,512), heads=(2,4,8,16)):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,dims[0],3,2,1), nn.BatchNorm2d(dims[0]), nn.SiLU())
        self.stage1=MaxViTBlock(dims[0],heads[0],s=1)
        self.down1=nn.Conv2d(dims[0],dims[1],3,2,1)
        self.stage2=MaxViTBlock(dims[1],heads[1])
        self.down2=nn.Conv2d(dims[1],dims[2],3,2,1)
        self.stage3=MaxViTBlock(dims[2],heads[2])
        self.down3=nn.Conv2d(dims[2],dims[3],3,2,1)
        self.stage4=MaxViTBlock(dims[3],heads[3])
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dims[3], num_classes))
    def forward(self,x):
        x=self.stem(x); x=self.stage1(x); x=self.down1(x); x=self.stage2(x); x=self.down2(x); x=self.stage3(x); x=self.down3(x); x=self.stage4(x)
        return self.head(x)
