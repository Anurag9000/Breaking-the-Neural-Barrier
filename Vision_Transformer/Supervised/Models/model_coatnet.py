import torch
import torch.nn as nn

# --------------------------------------------------
# CoAtNet (simplified): conv stages (MBConv) then attention stages (ViT blocks)
# --------------------------------------------------

class MBConv(nn.Module):
    def __init__(self, in_ch, out_ch, s=1, exp=4):
        super().__init__()
        hid = in_ch*exp
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, 1, 0, bias=False), nn.BatchNorm2d(hid), nn.SiLU(),
            nn.Conv2d(hid, hid, 3, s, 1, groups=hid, bias=False), nn.BatchNorm2d(hid), nn.SiLU(),
            nn.Conv2d(hid, out_ch, 1, 1, 0, bias=False), nn.BatchNorm2d(out_ch)
        )
        self.down = s>1 or in_ch!=out_ch
        if self.down:
            self.skip = nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, s, 0, bias=False), nn.BatchNorm2d(out_ch))
    def forward(self,x):
        y=self.block(x)
        if hasattr(self,'skip'): x=self.skip(x)
        return torch.relu(x+y)

class PatchEmbed(nn.Module):
    def __init__(self, in_ch, dim, patch):
        super().__init__()
        self.proj=nn.Conv2d(in_ch, dim, patch, patch)
        self.norm=nn.LayerNorm(dim)
    def forward(self,x):
        x=self.proj(x); B,C,H,W=x.shape
        x=x.flatten(2).transpose(1,2); x=self.norm(x); return x,H,W

class MLP(nn.Module):
    def __init__(self, dim, r=4.0):
        super().__init__()
        hid=int(dim*r); self.fc1=nn.Linear(dim,hid); self.act=nn.GELU(); self.fc2=nn.Linear(hid,dim)
    def forward(self,x): return self.fc2(self.act(self.fc1(x)))

class Attn(nn.Module):
    def __init__(self, dim, heads):
        super().__init__(); self.h=heads; self.scale=(dim//heads)**-0.5
        self.qkv=nn.Linear(dim,dim*3); self.proj=nn.Linear(dim,dim)
    def forward(self,x):
        B,N,C=x.shape
        qkv=self.qkv(x).reshape(B,N,3,self.h,C//self.h).permute(2,0,3,1,4)
        q,k,v=qkv[0],qkv[1],qkv[2]
        a=(q@k.transpose(-2,-1))*self.scale; a=a.softmax(-1)
        o=(a@v).transpose(1,2).reshape(B,N,C)
        return self.proj(o)

class TransformerStage(nn.Module):
    def __init__(self, dim, depth, heads):
        super().__init__(); self.blocks=nn.ModuleList([nn.ModuleList([nn.LayerNorm(dim), Attn(dim,heads), nn.LayerNorm(dim), MLP(dim)]) for _ in range(depth)])
    def forward(self,x):
        for n1,a,n2,m in self.blocks:
            x=x+a(n1(x)); x=x+m(n2(x))
        return x

class CoAtNet(nn.Module):
    def __init__(self, num_classes=10, dims=(64,128,256,512), attn_depth=(0,0,2,2), attn_heads=(0,0,4,8)):
        super().__init__()
        # Conv stages
        self.conv1 = MBConv(3, dims[0], s=2)
        self.conv2 = MBConv(dims[0], dims[1], s=2)
        # Attention stages
        self.pe3 = PatchEmbed(dims[1], dims[2], patch=2)
        self.tr3 = TransformerStage(dims[2], attn_depth[2], attn_heads[2])
        self.pe4 = PatchEmbed(dims[2], dims[3], patch=2)
        self.tr4 = TransformerStage(dims[3], attn_depth[3], attn_heads[3])
        self.head = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(dims[1], num_classes))
        self.head_attn = nn.Linear(dims[3], num_classes)
    def forward(self,x):
        x=self.conv1(x); x=self.conv2(x)
        pooled = nn.functional.adaptive_avg_pool2d(x,1).flatten(1)
        # attention path
        z,H,W=self.pe3(x); z=self.tr3(z); z,H,W=self.pe4(z.transpose(1,2).view(x.size(0),-1,H,W)); z=self.tr4(z)
        z=z.mean(1)
        # simple fusion: average logits
        logits_conv = self.head[2](pooled)
        logits_attn = self.head_attn(z)
        return 0.5*(logits_conv + logits_attn)
