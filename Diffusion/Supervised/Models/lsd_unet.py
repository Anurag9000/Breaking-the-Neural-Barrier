import torch.nn as nn
from core.diffusion_core import SinusoidalTimeEmbedding
from models.unet_blocks import ResBlock


class LatentUNet(nn.Module):
    def __init__(self, z_ch=4, base=128, tdim=256):
        super().__init__()

        # Time embedding
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(tdim),
            nn.SiLU()
        )

        # Encoder
        self.inp = nn.Conv2d(z_ch, base, 3, padding=1)
        self.d1 = ResBlock(base, tdim, base)
        self.d2 = ResBlock(base, tdim, base * 2)
        self.d3 = ResBlock(base * 2, tdim, base * 4)

        # Middle block
        self.mid = ResBlock(base * 4, tdim)

        # Decoder
        self.u3 = ResBlock(base * 4, tdim, base * 2)
        self.u2 = ResBlock(base * 2, tdim, base)
        self.u1 = ResBlock(base, tdim, base)

        # Output
        self.out = nn.Conv2d(base, z_ch, 3, padding=1)

    def forward(self, zt, t, _):
        t_emb = self.time(t)

        # Downsampling path
        x0 = self.inp(zt)
        d1 = self.d1(x0, t_emb)
        d2 = self.d2(d1, t_emb)
        d3 = self.d3(d2, t_emb)

        # Bottleneck
        m = self.mid(d3, t_emb)

        # Upsampling path
        u3 = self.u3(m, t_emb)
        u2 = self.u2(u3, t_emb)
        u1 = self.u1(u2, t_emb)

        # Output
        return self.out(u1)
