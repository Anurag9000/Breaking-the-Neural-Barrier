import torch
import torch.nn as nn

# ------------------------------
# PiT (Pooling-based Vision Transformer)
# - ViT blocks with token pooling between stages (reduces tokens, increases dim)
# ------------------------------

from model_vit import EncoderBlock

class TokenPool(nn.Module):
    def __init__(self, dim, out_dim, pool=2, img_size=224, patch=16):
        super().__init__()
        self.pool=pool
        self.proj=nn.Linear(dim*pool*pool, out_dim)
        self.img_size=img_size; self.patch=patch
    def forward(self, x):  # x: (B, N+1, C) with CLS at 0
        cls, tok = x[:, :1], x[:, 1:]
        B,N,C = tok.shape
        S = int((N)**0.5)
        tok = tok.view(B,S,S,C)
        # 2x2 non-overlap pool
        S2 = S//self.pool
        tok = tok[:, :S2*self.pool, :S2*self.pool, :]
        tok = tok.view(B, S2, self.pool, S2, self.pool, C).permute(0,1,3,2,4,5).contiguous()
        tok = tok.reshape(B, S2, S2, C*self.pool*self.pool).reshape(B, S2*S2, C*self.pool*self.pool)
        tok = self.proj(tok)
        return torch.cat([cls, tok], dim=1)

class PiT(nn.Module):
    def __init__(self, img_size=224, patch=16, num_classes=10, dims=(192, 256, 320), depths=(4,4,4), heads=(3,4,5)):
        super().__init__()
        # patch embed via linear proj (from model_vit.PatchEmbed)
        from model_vit import PatchEmbed
        self.patch = PatchEmbed(img_size, patch, 3, dims[0])
        self.cls = nn.Parameter(torch.zeros(1,1,dims[0]))
        self.pos = None
        # stages
        self.stages = nn.ModuleList()
        dim=dims[0]
        for i in range(len(depths)):
            blocks = nn.ModuleList([EncoderBlock(dim, heads[i], mlp_ratio=4.0) for _ in range(depths[i])])
            self.stages.append(blocks)
            if i < len(depths)-1:
                self.stages.append(TokenPool(dim, dims[i+1], pool=2, img_size=img_size, patch=patch))
                dim=dims[i+1]
        self.norm=nn.LayerNorm(dim)
        self.head=nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)

    def forward(self,x):
        B=x.size(0)
        tok=self.patch(x)  # (B,N,C)
        N=tok.size(1)
        if self.pos is None or self.pos.size(1)!=(N+1):
            self.pos=nn.Parameter(torch.zeros(1,N+1,tok.size(2), device=tok.device))
            nn.init.trunc_normal_(self.pos, std=0.02)
        cls=self.cls.expand(B,-1,-1)
        z=torch.cat([cls,tok],dim=1)
        z=z+self.pos
        for m in self.stages:
            if isinstance(m, nn.ModuleList):
                for blk in m: z=blk(z)
            else:
                z=m(z)
        z=self.norm(z)
        return self.head(z[:,0])
