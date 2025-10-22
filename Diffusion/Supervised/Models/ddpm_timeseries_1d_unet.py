import torch.nn as nn
from models.unet_blocks import SimpleUNet


class TS1DUNet(SimpleUNet):
    """
    1D Time-Series Denoising UNet using a 2D UNet internally.
    Input shape: (batch, ch, T) -> reshaped to (batch, ch, 1, T) internally.
    """

    def __init__(self, ch=1, base=64, tdim=256):
        # Use 2D UNet by reshaping (ch, T) -> (ch, 1, T)
        super().__init__(in_ch=ch, out_ch=ch, base=base, tdim=tdim, cond_dim=0)

    def forward(self, y_noisy, t, _ctx=None):
        # Pass through UNet; no conditioning
        return super().forward(y_noisy, t, None)
