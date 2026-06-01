import torch
import torch.nn as nn

# --------------------------------------------------
# CaiT (Class-Attention in Image Transformers) - simplified
#  - Standard ViT encoder blocks
#  - Followed by K class-attention blocks operating on CLS + tokens
# --------------------------------------------------

from model_vit import PatchEmbed, EncoderBlock

class ClassAttentionBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.proj_drop = nn.Dropout(proj_drop)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Dropout(drop), nn.Linear(int(dim*mlp_ratio), dim), nn.Dropout(drop))
    def forward(self, x):
        # x: (B, 1+N, D) ; class attends to full tokens, tokens unchanged
        cls, tok = x[:, :1], x[:, 1:]
        q = self.norm1(cls)
        k = self.norm1(x)
        cls2, _ = self.attn(q, k, k, need_weights=False)
        cls = cls + self.proj_drop(cls2)
        x = torch.cat([cls, tok], dim=1)
        x = x + self.mlp(self.norm2(x))
        return x

class CaiT(nn.Module):
    def __init__(self, img_size=224, patch=16, num_classes=10, dim=384, depth=12, heads=6, ca_blocks=2, mlp_ratio=4.0):
        super().__init__()
        self.patch = PatchEmbed(img_size, patch, 3, dim)
        self.cls = nn.Parameter(torch.zeros(1,1,dim))
        self.pos = None
        self.enc = nn.ModuleList([EncoderBlock(dim, heads, mlp_ratio) for _ in range(depth)])
        self.ca = nn.ModuleList([ClassAttentionBlock(dim, heads, mlp_ratio) for _ in range(ca_blocks)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
        nn.init.trunc_normal_(self.cls, std=0.02)
    def forward(self,x):
        B=x.size(0)
        t=self.patch(x); N=t.size(1)
        if self.pos is None or self.pos.size(1)!=(N+1):
            self.pos=nn.Parameter(torch.zeros(1,N+1,t.size(2),device=t.device)); nn.init.trunc_normal_(self.pos, std=0.02)
        z=torch.cat([self.cls.expand(B,-1,-1), t], dim=1)
        z=z+self.pos
        for b in self.enc: z=b(z)
        for b in self.ca: z=b(z)
        z=self.norm(z)
        return self.head(z[:,0])
