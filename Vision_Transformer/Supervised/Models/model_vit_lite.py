import torch
import torch.nn as nn

# ------------------------------
# ViT-Lite (compact ViT for small images / mobile)
# - Same ViT encoder, smaller dims, smaller patches
# - Kept canonical (CLS token + pos embed, Pre-LN)
# ------------------------------

from model_vit import PatchEmbed, EncoderBlock

class ViT_Lite(nn.Module):
    def __init__(self, img_size=224, patch_size=8, in_chans=3, num_classes=10,
                 embed_dim=192, depth=12, num_heads=3, mlp_ratio=3.0, drop=0.0, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.patch_embed.num_patches
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(drop)
        self.blocks = nn.ModuleList([
            EncoderBlock(embed_dim, num_heads, mlp_ratio, attn_drop, proj_drop, drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        if self.head.bias is not None:
            nn.init.zeros_(self.head.bias)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return self.head(x[:, 0])
