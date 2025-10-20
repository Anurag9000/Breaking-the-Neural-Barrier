import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, img=128, patch=8, dim=256):
        super().__init__(); self.grid=(img//patch, img//patch); self.dim=dim
        self.proj = nn.Conv2d(in_ch, dim, patch, patch)
    def forward(self, x):
        x = self.proj(x); B,C,H,W = x.shape
        return x, H, W

class TransformerEncoder(nn.Module):
    def __init__(self, dim=256, depth=4, nhead=8):
        super().__init__()
        enc = nn.TransformerEncoderLayer(dim, nhead, dim*4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
    def forward(self, x):
        B,C,H,W = x.shape
        z = x.flatten(2).transpose(1,2)
        z = self.encoder(z)
        return z.transpose(1,2).view(B,C,H,W)

class TransUNet(nn.Module):
    """U-Net with Transformer bottleneck (single-model)."""
    def __init__(self, num_classes=3, img=128, patch=8, dim=128):
        super().__init__()
        self.enc1 = nn.Sequential(nn.Conv2d(3, dim, 3, 1, 1), nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU())
        self.down1 = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(nn.Conv2d(dim, dim*2, 3, 1, 1), nn.ReLU(), nn.Conv2d(dim*2, dim*2, 3, 1, 1), nn.ReLU())
        self.down2 = nn.MaxPool2d(2)
        self.trans = TransformerEncoder(dim*2, depth=4, nhead=8)
        self.up1 = nn.ConvTranspose2d(dim*2, dim, 2, 2)
        self.dec1 = nn.Sequential(nn.Conv2d(dim*2, dim, 3, 1, 1), nn.ReLU(), nn.Conv2d(dim, dim, 3, 1, 1), nn.ReLU())
        self.up2 = nn.ConvTranspose2d(dim, dim//2, 2, 2)
        self.dec2 = nn.Sequential(nn.Conv2d(dim//2+3, dim//2, 3, 1, 1), nn.ReLU(), nn.Conv2d(dim//2, num_classes, 1))
    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        b = self.trans(self.down2(e2))
        u1 = self.up1(b)
        d1 = self.dec1(torch.cat([u1, e2], dim=1))
        u2 = self.up2(d1)
        out = self.dec2(torch.cat([u2, x], dim=1))
        return out
