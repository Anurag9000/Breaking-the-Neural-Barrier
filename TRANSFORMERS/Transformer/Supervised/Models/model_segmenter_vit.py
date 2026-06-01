import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img=128, patch=8, in_ch=3, dim=256):
        super().__init__(); self.grid=(img//patch, img//patch); self.dim=dim
        self.proj = nn.Conv2d(in_ch, dim, patch, patch)
    def forward(self, x):
        x = self.proj(x)
        B,C,H,W = x.shape
        return x.flatten(2).transpose(1,2), H, W

class SegmenterViT(nn.Module):
    """Segmenter-style: ViT encoder + lightweight pixel decoder.
    Output: per-pixel class logits, upsampled to input size.
    """
    def __init__(self, num_classes=3, img=128, patch=8, dim=256, depth=6, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.patch = PatchEmbed(img, patch, 3, dim)
        self.pos = nn.Parameter(torch.zeros(1, (img//patch)*(img//patch), dim))
        enc = nn.TransformerEncoderLayer(dim, nhead, int(dim*mlp_ratio), 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, num_classes)
        )
        self.patch_size = patch; self.img = img
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        B, C, H, W = x.shape
        z, h, w = self.patch(x)
        z = self.encoder(z + self.pos[:, : z.size(1)])
        logits = self.decoder(z)  # B, N, K
        logits = logits.transpose(1,2).view(B, -1, h, w)
        logits = F.interpolate(logits, size=(H,W), mode='bilinear', align_corners=False)
        return logits
