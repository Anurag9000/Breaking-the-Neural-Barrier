import torch
import torch.nn as nn

class MV2Block(nn.Module):
    def __init__(self, in_ch, out_ch, expand=4, stride=1):
        super().__init__()
        mid = in_ch*expand
        self.use_res = (in_ch==out_ch and stride==1)
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, mid, 1, 1, 0), nn.BatchNorm2d(mid), nn.SiLU(),
            nn.Conv2d(mid, mid, 3, stride, 1, groups=mid), nn.BatchNorm2d(mid), nn.SiLU(),
            nn.Conv2d(mid, out_ch, 1, 1, 0), nn.BatchNorm2d(out_ch)
        )
    def forward(self, x):
        y = self.net(x)
        return x + y if self.use_res else y

class TransformerBlock(nn.Module):
    def __init__(self, dim, nhead=4):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, 4*dim), nn.SiLU(), nn.Linear(4*dim, dim))
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x), need_weights=False)[0]
        x = x + self.mlp(self.ln2(x))
        return x

class MobileViTBlock(nn.Module):
    def __init__(self, in_ch, dim=144, nhead=4, depth=2):
        super().__init__()
        self.local = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, 3, 1, 1), nn.SiLU(), nn.Conv2d(in_ch, dim, 1)
        )
        self.trans = nn.ModuleList([TransformerBlock(dim, nhead) for _ in range(depth)])
        self.proj = nn.Conv2d(dim, in_ch, 1)
    def forward(self, x):
        y = self.local(x)  # B, D, H, W
        B,D,H,W = y.shape
        z = y.flatten(2).transpose(1,2)
        for blk in self.trans: z = blk(z)
        z = z.transpose(1,2).view(B,D,H,W)
        z = self.proj(z)
        return torch.cat([x, z], dim=1)

class MobileViT(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, 1), nn.BatchNorm2d(16), nn.SiLU(),
            MV2Block(16, 32, stride=2),
        )
        self.block1 = MobileViTBlock(32, dim=96)
        self.down1 = nn.Conv2d(32+32, 64, 3, 2, 1)
        self.block2 = MobileViTBlock(64, dim=128)
        self.down2 = nn.Conv2d(64+64, 128, 3, 2, 1)
        self.block3 = MobileViTBlock(128, dim=160)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(128+128, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.pool(x).flatten(1)
        return self.head(x)
