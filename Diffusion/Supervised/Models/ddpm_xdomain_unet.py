import torch
import torch.nn as nn
from models.unet_blocks import SimpleUNet


class XDomainUNet(nn.Module):
    def __init__(self, y_ch=1, x_ch=1, base=64, tdim=256):
        super().__init__()
        # Denoiser sees (noisy target Y + source X) channels
        self.net = SimpleUNet(
            in_ch=y_ch + x_ch, 
            out_ch=y_ch, 
            base=base, 
            tdim=tdim, 
            cond_dim=0
        )

    def forward(self, y_t, t, x_src):
        # Concatenate noisy target and source channels along channel dimension
        return self.net(torch.cat([y_t, x_src], dim=1), t, None)
