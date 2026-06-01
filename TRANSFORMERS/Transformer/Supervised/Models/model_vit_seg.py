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

class ViTSeg(nn.Module):
    def __init__(self, num_classes=3, img=128, patch=8, dim=256, depth=6, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.patch = PatchEmbed(img, patch, 3, dim)
        self.pos = nn.Parameter(torch.zeros(1, (img//patch)*(img//patch), dim))
        enc = nn.TransformerEncoderLayer(dim, nhead, int(dim*mlp_ratio), 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(dim, dim//2, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(dim//2, dim//4, kernel_size=2, stride=2), nn.ReLU(inplace=True),
            nn.Conv2d(dim//4, num_classes, kernel_size=1)
        )
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.patch_size = patch

    def forward(self, x):
        B,C,H,W = x.shape
        z, h, w = self.patch(x)
        z = self.encoder(z + self.pos[:, : z.size(1)])  # B, N, D
        z = z.transpose(1,2).view(B, -1, h, w)
        scale = self.patch_size
        # upsample by conv transpose layers to reach input size (assumes patch=8, two upsamples -> factor 4; final bilinear to exact H,W)
        y = self.decoder(z)
        y = F.interpolate(y, size=(H,W), mode='bilinear', align_corners=False)
        return y
