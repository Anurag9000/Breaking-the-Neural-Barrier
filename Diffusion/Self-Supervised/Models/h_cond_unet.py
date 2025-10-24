# models/h_cond_unet.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.resblock import ResBlock  # assuming you have a ResBlock implemented
from models.utils import time_embedding  # assuming a time embedding function

# -------------------------
# Down-sampling block
# -------------------------
class Down(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)
        self.down = nn.Conv2d(c_out, c_out, 3, stride=2, padding=1)

    def forward(self, x, c):
        x = self.block(x, c)
        return self.down(x)


# -------------------------
# Up-sampling block
# -------------------------
class Up(nn.Module):
    def __init__(self, c_in, c_out, cond_dim):
        super().__init__()
        self.block = ResBlock(c_in, c_out, cond_dim)

    def forward(self, x, skip, c):
        x = F.interpolate(x, scale_factor=2, mode='nearest')
        x = torch.cat([x, skip], dim=1)
        return self.block(x, c)


# -------------------------
# Conditional UNet
# -------------------------
class CondUNet(nn.Module):
    def __init__(self, base=64, in_ch=3, cond_ch=0, sc_ch=0, out_ch=3,
                 cond_dim=256, num_classes=None, cfg_null=False):
        super().__init__()
        self.cond_dim = cond_dim
        self.num_classes = num_classes
        self.cfg_null = cfg_null

        extra = 0
        if num_classes is not None:
            self.class_emb = nn.Embedding(num_classes + (1 if cfg_null else 0), cond_dim)
            extra += cond_dim

        self.cond_proj = nn.Sequential(
            nn.Linear(cond_dim + extra, cond_dim),
            nn.SiLU(),
            nn.Linear(cond_dim, cond_dim)
        )

        total_in = in_ch + cond_ch + sc_ch
        self.enc1 = ResBlock(total_in, base, cond_dim)
        self.down1 = Down(base, base*2, cond_dim)
        self.down2 = Down(base*2, base*4, cond_dim)
        self.mid = ResBlock(base*4, base*4, cond_dim)
        self.up1 = Up(base*8, base*2, cond_dim)
        self.up2 = Up(base*4, base, cond_dim)
        self.out = nn.Conv2d(base*2, out_ch, 3, padding=1)

    def forward(self, x_in, t, y=None):
        # x_in already concatenates [x_t, conditioning, optional self-cond]
        t_emb = time_embedding(t, self.cond_dim)

        if self.num_classes is not None:
            cls = self.class_emb(y)  # y may already be null-id for CFG unconditional
            t_emb = torch.cat([t_emb, cls], dim=1)

        c = self.cond_proj(t_emb)

        s1 = self.enc1(x_in, c)
        x = self.down1(s1, c)
        s2 = x
        x = self.down2(x, c)
        x = self.mid(x, c)
        x = self.up1(x, s2, c)
        x = self.up2(x, s1, c)
        return self.out(x)
