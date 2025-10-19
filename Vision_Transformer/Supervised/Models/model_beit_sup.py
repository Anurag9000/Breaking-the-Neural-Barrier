import torch
import torch.nn as nn

# --------------------------------------------------
# BEiT (supervised encoder variant):
#  - Use a ViT-like encoder, trained supervised end-to-end.
#  - No tokenizer/EMA; plain single-model classifier.
# --------------------------------------------------

from model_vit import PatchEmbed, EncoderBlock

class BEiT_Sup(nn.Module):
    def __init__(self, img_size=224, patch=16, num_classes=10, dim=768, depth=12, heads=12, mlp_ratio=4.0):
        super().__init__()
        self.patch=PatchEmbed(img_size, patch, 3, dim)
        self.cls=nn.Parameter(torch.zeros(1,1,dim))
        self.pos=None
        self.blocks=nn.ModuleList([EncoderBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self,x):
        B=x.size(0)
        t=self.patch(x); N=t.size(1)
        if self.pos is None or self.pos.size(1)!=(N+1):
            self.pos=nn.Parameter(torch.zeros(1,N+1,t.size(2), device=t.device)); nn.init.trunc_normal_(self.pos, std=0.02)
        z=torch.cat([self.cls.expand(B,-1,-1), t], dim=1); z=z+self.pos
        for b in self.blocks: z=b(z)
        z=self.norm(z)
        return self.head(z[:,0])
