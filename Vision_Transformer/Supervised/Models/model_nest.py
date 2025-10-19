import torch
import torch.nn as nn

# --------------------------------------------------
# NesT (simplified): nested transformer with block-wise processing and merge
# --------------------------------------------------

from model_vit import EncoderBlock

class BlockMerge(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(4*dim, 2*dim)
    def forward(self, x, H, W):
        # x: (B, H*W, C)
        B,N,C=x.shape
        x=x.view(B,H,W,C)
        x=torch.cat([x[:,0::2,0::2,:], x[:,0::2,1::2,:], x[:,1::2,0::2,:], x[:,1::2,1::2,:]], dim=-1)
        x=x.view(B,-1,4*C)
        x=self.proj(x)
        return x, H//2, W//2

class NesT(nn.Module):
    def __init__(self, img_size=224, patch=4, num_classes=10, dims=(96,192,384), depths=(2,2,6), heads=(3,6,12)):
        super().__init__()
        from model_vit import PatchEmbed
        self.patch = PatchEmbed(img_size, patch, 3, dims[0])
        self.stages = nn.ModuleList()
        dim=dims[0]
        H=W=None
        for i in range(len(depths)):
            blks = nn.ModuleList([EncoderBlock(dim, heads[i], 4.0) for _ in range(depths[i])])
            self.stages.append(blks)
            if i < len(depths)-1:
                self.stages.append(BlockMerge(dim))
                dim = dims[i+1]
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim, num_classes)
    def forward(self,x):
        x,H,W=self.patch(x)
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for b in m: x=b(x)
            else:
                x,H,W=m(x,H,W)
        x=self.norm(x); x=x.mean(1)
        return self.head(x)
