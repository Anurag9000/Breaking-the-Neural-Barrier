import torch
import torch.nn as nn
from models.unet_blocks import SimpleUNet


class CtrlDepthUNet(nn.Module):
    """U-Net that conditions image denoising on a depth map."""

    def __init__(self, img_ch=3, depth_ch=1, base=64, tdim=256):
        super().__init__()
        # Input to denoiser is (noisy RGB + depth map)
        self.net = SimpleUNet(
            in_ch=img_ch + depth_ch,
            out_ch=img_ch,
            base=base,
            tdim=tdim,
            cond_dim=0
        )

    def forward(self, x_t, t, depth):
        # Concatenate the noisy image and depth along the channel dimension
        x = torch.cat([x_t, depth], dim=1)
        return self.net(x, t, None)
