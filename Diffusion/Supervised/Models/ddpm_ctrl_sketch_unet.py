import torch
import torch.nn as nn
from models.unet_blocks import SimpleUNet


class CtrlSketchUNet(nn.Module):
    """U-Net that conditions image denoising on an input sketch."""

    def __init__(self, img_ch=3, sketch_ch=1, base=64, tdim=256):
        super().__init__()
        # Input to denoiser is (noisy RGB + sketch)
        self.net = SimpleUNet(
            in_ch=img_ch + sketch_ch,
            out_ch=img_ch,
            base=base,
            tdim=tdim,
            cond_dim=0,
        )

    def forward(self, x_t, t, sketch):
        # Concatenate the noisy image and sketch along the channel dimension
        x = torch.cat([x_t, sketch], dim=1)
        return self.net(x, t, None)
