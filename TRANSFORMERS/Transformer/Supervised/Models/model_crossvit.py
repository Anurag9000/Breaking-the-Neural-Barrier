import torch
import torch.nn as nn

# --------------------------------------------------
# CrossViT (simplified, single-model):
#  - Two branches with different patch sizes (small & large)
#  - Fuse CLS tokens via concatenation and projection
# --------------------------------------------------

from model_vit import PatchEmbed, EncoderBlock

class CrossViT(nn.Module):
    def __init__(self, img_size=224, num_classes=10,
                 s_dim=192, s_patch=8, s_depth=6, s_heads=3,
                 l_dim=384, l_patch=16, l_depth=6, l_heads=6):
        super().__init__()
        # small branch
        self.s_patch = PatchEmbed(img_size, s_patch, 3, s_dim)
        self.s_cls = nn.Parameter(torch.zeros(1,1,s_dim))
        self.s_pos = None
        self.s_blocks = nn.ModuleList([EncoderBlock(s_dim, s_heads, 4.0) for _ in range(s_depth)])
        # large branch
        self.l_patch = PatchEmbed(img_size, l_patch, 3, l_dim)
        self.l_cls = nn.Parameter(torch.zeros(1,1,l_dim))
        self.l_pos = None
        self.l_blocks = nn.ModuleList([EncoderBlock(l_dim, l_heads, 4.0) for _ in range(l_depth)])
        # fusion
        self.fuse = nn.Linear(s_dim + l_dim, l_dim)
        self.norm = nn.LayerNorm(l_dim)
        self.head = nn.Linear(l_dim, num_classes)
        nn.init.trunc_normal_(self.s_cls, std=0.02)
        nn.init.trunc_normal_(self.l_cls, std=0.02)
    def forward(self,x):
        B=x.size(0)
        # small
        ts=self.s_patch(x); Ns=ts.size(1)
        if self.s_pos is None or self.s_pos.size(1)!=(Ns+1):
            self.s_pos=nn.Parameter(torch.zeros(1,Ns+1,ts.size(2), device=ts.device)); nn.init.trunc_normal_(self.s_pos,std=0.02)
        zs=torch.cat([self.s_cls.expand(B,-1,-1), ts], dim=1); zs=zs+self.s_pos
        for b in self.s_blocks: zs=b(zs)
        # large
        tl=self.l_patch(x); Nl=tl.size(1)
        if self.l_pos is None or self.l_pos.size(1)!=(Nl+1):
            self.l_pos=nn.Parameter(torch.zeros(1,Nl+1,tl.size(2), device=tl.device)); nn.init.trunc_normal_(self.l_pos,std=0.02)
        zl=torch.cat([self.l_cls.expand(B,-1,-1), tl], dim=1); zl=zl+self.l_pos
        for b in self.l_blocks: zl=b(zl)
        # fuse CLS
        cls = torch.cat([zs[:,0], zl[:,0]], dim=-1)
        cls = self.fuse(cls)
        cls = self.norm(cls)
        return self.head(cls)
