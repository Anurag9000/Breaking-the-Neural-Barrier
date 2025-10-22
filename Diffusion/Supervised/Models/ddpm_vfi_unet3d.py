import torch
import torch.nn as nn
from models.unet_blocks import SimpleUNet


# Minimal proxy using 2D UNet on concatenated pair + time‑pos
# Practical VFI would typically use 3D UNet for better temporal modeling
class VFIUNet(nn.Module):
    def __init__(self, img_ch=3, base=64, tdim=256):
        super().__init__()
        # Input: mid frame noisy + previous + next frame -> 3*img_ch channels
        self.net = SimpleUNet(
            in_ch=img_ch * 2 + img_ch,  # mid_noisy + prev + next
            out_ch=img_ch,
            base=base,
            tdim=tdim,
            cond_dim=0
        )

    def forward(self, mid_noisy, t, ctx):
        prev, nxt = ctx  # ctx = tuple(prev_frame, next_frame)
        x = torch.cat([mid_noisy, prev, nxt], dim=1)  # concatenate along channel dim
        return self.net(x, t, None)
