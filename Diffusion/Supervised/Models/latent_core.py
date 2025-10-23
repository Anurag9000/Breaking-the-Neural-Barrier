import torch
import torch.nn as nn
import torch.nn.functional as F


class EncBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU()
        )
        self.down = nn.Conv2d(c_out, c_out, 4, stride=2, padding=1)

    def forward(self, x):
        x = self.conv(x)
        skip = x
        x = self.down(x)
        return x, skip


class DecBlock(nn.Module):
    def __init__(self, c_in, c_skip, c_out):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c_in + c_skip, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1),
            nn.GroupNorm(8, c_out),
            nn.SiLU()
        )

    def forward(self, x, skip):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class DeterministicAutoencoder(nn.Module):
    def __init__(self, img_ch=3, base=64, z_ch=4):
        super().__init__()
        # Encoder
        self.e1 = EncBlock(img_ch, base)
        self.e2 = EncBlock(base, base * 2)
        self.e3 = EncBlock(base * 2, base * 4)

        # Bottleneck
        self.to_z = nn.Conv2d(base * 4, z_ch, 3, padding=1)
        self.from_z = nn.Conv2d(z_ch, base * 4, 3, padding=1)

        # Decoder
        self.d3 = DecBlock(base * 4, base * 4, base * 2)
        self.d2 = DecBlock(base * 2, base * 2, base)
        self.d1 = DecBlock(base, base, base)
        self.out = nn.Conv2d(base, img_ch, 3, padding=1)

    def encode(self, x):
        x, s1 = self.e1(x)
        x, s2 = self.e2(x)
        x, s3 = self.e3(x)
        z = self.to_z(x)
        return z, (s1, s2, s3)

    def decode(self, z, skips):
        s1, s2, s3 = skips
        x = self.from_z(z)
        x = self.d3(x, s3)
        x = self.d2(x, s2)
        x = self.d1(x, s1)
        return self.out(x)
