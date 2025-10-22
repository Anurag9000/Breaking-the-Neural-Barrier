import torch
import torch.nn as nn
from models.unet_blocks import SimpleUNet


# Proxy: treat depth slices as channels; real use should implement true 3D UNet.
class Med3DUNet(nn.Module):
    def __init__(self, vol_ch=8, out_ch=1, base=64, tdim=256):
        super().__init__()
        # input: noisy volume + original volume as context
        self.net = SimpleUNet(in_ch=vol_ch + out_ch, out_ch=out_ch, base=base, tdim=tdim, cond_dim=0)

    def forward(self, y_noisy, t, vol):
        # concatenate along channel dimension: [B, vol_ch + out_ch, H, W]
        x = torch.cat([y_noisy, vol], dim=1)
        return self.net(x, t, None)
