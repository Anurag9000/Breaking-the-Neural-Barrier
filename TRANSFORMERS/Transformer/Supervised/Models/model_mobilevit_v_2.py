import torch
import torch.nn as nn

# --------------------------------------------------
# MobileViT v2 (simplified): conv bottlenecks + transformer block over unfolded features
# --------------------------------------------------

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, s=1, exp=4):
        super().__init__(); hid=in_ch*exp
        self.block=nn.Sequential(
            nn.Conv2d(in_ch,hid,1,1,0,bias=False), nn.BatchNorm2d(hid), nn.SiLU(),
            nn.Conv2d(hid,hid,3,s,1,groups=hid,bias=False), nn.BatchNorm2d(hid), nn.SiLU(),
            nn.Conv2d(hid,out_ch,1,1,0,bias=False), nn.BatchNorm2d(out_ch)
        )
        self.res = (s==1 and in_ch==out_ch)
    def forward(self,x): y=self.block(x); return x+y if self.res else y

class Transformer(nn.Module):
    def __init__(self, dim, heads=4, depth=2):
        super().__init__(); self.blocks=nn.ModuleList()
        for _ in range(depth):
            self.blocks.append(nn.ModuleList([nn.LayerNorm(dim), nn.MultiheadAttention(dim, heads, batch_first=True), nn.LayerNorm(dim), nn.Sequential(nn.Linear(dim,dim*4), nn.GELU(), nn.Linear(dim,dim))]))
    def forward(self,t):
        for n1,a,n2,m in self.blocks:
            z=n1(t); z,_=a(z,z,z, need_weights=False); t=t+z; t=t+m(n2(t))
        return t

class MobileViTBlock(nn.Module):
    def __init__(self, c, heads=4, depth=2, patch=2):
        super().__init__(); self.patch=patch; self.lin1=nn.Conv2d(c,c,1); self.tr=Transformer(c,heads,depth); self.lin2=nn.Conv2d(c,c,1)
    def forward(self,x):
        B,C,H,W=x.shape
        y=self.lin1(x)
        # unfold non-overlapping patches (p x p)
        p=self.patch; Hp,Hw=H//p,W//p; y=y[:,:,:Hp*p,:Hw*p]
        t=y.view(B,C,Hp,p,Hw,p).permute(0,2,1,3,5,4).contiguous().view(B, C, p*p, Hp*Hw).permute(0,3,2,1).reshape(B, Hp*Hw, p*p*C)
        # project to dim C
        proj_in=nn.Linear(p*p*C, C).to(y.device)
        t=proj_in(t)
        t=self.tr(t)
        proj_out=nn.Linear(C, p*p*C).to(y.device)
        t=proj_out(t).view(B, Hp, Hw, p*p*C).permute(0,3,1,2).contiguous().view(B,C*p*p,Hp,Hw)
        y = torch.nn.functional.fold(t.view(B, C*p*p, Hp*Hw), output_size=(Hp*p, Hw*p), kernel_size=p, stride=p)
        y=self.lin2(y)
        # fuse
        return x + y

class MobileViT_v2(nn.Module):
    def __init__(self, num_classes=10, dims=(64,96,128,160)):
        super().__init__()
        self.stem=nn.Sequential(nn.Conv2d(3,dims[0],3,2,1), nn.BatchNorm2d(dims[0]), nn.SiLU())
        self.stage1=MBConv(dims[0],dims[0])
        self.down1=MBConv(dims[0],dims[1],s=2)
        self.mv2_1=MobileViTBlock(dims[1])
        self.down2=MBConv(dims[1],dims[2],s=2)
        self.mv2_2=MobileViTBlock(dims[2])
        self.down3=MBConv(dims[2],dims[3],s=2)
        self.mv2_3=MobileViTBlock(dims[3])
        self.head=nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dims[3], num_classes))
    def forward(self,x):
        x=self.stem(x); x=self.stage1(x); x=self.down1(x); x=self.mv2_1(x); x=self.down2(x); x=self.mv2_2(x); x=self.down3(x); x=self.mv2_3(x); return self.head(x)
