import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img=128, patch=8, in_ch=3, dim=256):
        super().__init__(); self.grid=(img//patch, img//patch)
        self.proj = nn.Conv2d(in_ch, dim, patch, patch)
    def forward(self, x):
        x = self.proj(x); B,C,H,W = x.shape
        return x.flatten(2).transpose(1,2), H, W

class Mask2FormerToy(nn.Module):
    """Simplified Mask2Former: ViT encoder -> fixed number of queries -> per-query class + mask via linear proj onto patch tokens.
    For toy shapes segmentation.
    """
    def __init__(self, num_classes=3, num_queries=6, img=128, patch=8, dim=256, depth=6, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.patch = PatchEmbed(img, patch, 3, dim)
        self.pos = nn.Parameter(torch.zeros(1, (img//patch)*(img//patch), dim))
        enc = nn.TransformerEncoderLayer(dim, nhead, int(dim*mlp_ratio), 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.query = nn.Parameter(torch.randn(1, num_queries, dim))
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.class_head = nn.Linear(dim, num_classes)
        self.mask_proj = nn.Linear(dim, dim)
        self.patch = PatchEmbed(img, patch, 3, dim)
        self.img = img; self.patch_size = patch; self.dim=dim

    def forward(self, x):
        B,C,H,W = x.shape
        z, h, w = self.patch(x)
        z = self.encoder(z + self.pos[:, : z.size(1)])  # B, N, D
        q = self.query.expand(B, -1, -1)
        q = self.q_proj(q)
        k = self.k_proj(z)
        v = self.v_proj(z)
        attn = (q @ k.transpose(1,2)) / (self.dim ** 0.5)
        attn = attn.softmax(-1)
        q_feat = attn @ v  # B, Q, D
        class_logits = self.class_head(q_feat)  # B, Q, K
        mask_embed = self.mask_proj(q_feat)     # B, Q, D
        # project masks by sim with patch tokens -> B, Q, N -> upsample to H,W
        masks = (mask_embed @ z.transpose(1,2))  # B, Q, N
        masks = masks.view(B, -1, h, w)
        masks = F.interpolate(masks, size=(H,W), mode='bilinear', align_corners=False)
        return class_logits, masks
