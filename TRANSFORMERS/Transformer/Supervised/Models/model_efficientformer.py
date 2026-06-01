import torch
import torch.nn as nn

# --------------------------------------------------
# EfficientFormer (simplified):
#  - Local MHRA-like conv-attn blocks for efficiency
#  - Hierarchical with depthwise conv token mixing + SE
# --------------------------------------------------

class SEModule(nn.Module):
    def __init__(self, c, r=4):
        super().__init__(); h=max(4,c//r)
        self.fc=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Conv2d(c,h,1), nn.ReLU(inplace=True), nn.Conv2d(h,c,1), nn.Sigmoid())
    def forward(self,x): return x*self.fc(x)

class MHRA(nn.Module):
    def __init__(self, c, heads=4):
        super().__init__(); self.h=heads; self.d=c//heads
        self.q=nn.Conv2d(c,c,1); self.kv=nn.Conv2d(c,2*c,1); self.proj=nn.Conv2d(c,c,1)
    def forward(self,x):
        B,C,H,W=x.shape
        q=self.q(x).view(B,self.h,self.d,H*W); kv=self.kv(x).view(B,2,self.h,self.d,H*W)
        k,v=kv[:,0],kv[:,1]
        a=(q.transpose(-2,-1)@k)/(self.d**0.5); a=a.softmax(-1)
        y=(a@v.transpose(-2,-1)).transpose(-2,-1).contiguous().view(B,C,H,W)
        return self.proj(y)

class EBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.dw=nn.Conv2d(c,c,3,1,1,groups=c)
        self.pw=nn.Conv2d(c,c,1)
        self.norm=nn.BatchNorm2d(c)
        self.act=nn.SiLU()
        self.mhra=MHRA(c)
        self.se=SEModule(c)
    def forward(self,x):
        y=self.act(self.norm(self.dw(x)))
        x=x+self.pw(y)
        x=x+self.se(self.mhra(x))
        return x

class Down(nn.Module):
    def __init__(self, ci, co): super().__init__(); self.c=nn.Conv2d(ci,co,3,2,1); self.n=nn.BatchNorm2d(co); self.a=nn.SiLU()
    def forward(self,x): return self.a(self.n(self.c(x)))

class EfficientFormer(nn.Module):
    def __init__(self, num_classes=10, dims=(64,128,256,512), depths=(2,2,6,2)):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,dims[0],3,2,1), nn.BatchNorm2d(dims[0]), nn.SiLU())
        stages=[]; c=dims[0]
        for i,d in enumerate(depths):
            blocks=[EBlock(c) for _ in range(d)]; stages+=blocks
            if i<3:
                stages.append(Down(c,dims[i+1])); c=dims[i+1]
        self.stages=nn.Sequential(*stages)
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dims[3], num_classes))
    def forward(self,x): x=self.stem(x); x=self.stages(x); return self.head(x)
