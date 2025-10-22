import torch
import torch.nn as nn
from models.unet_blocks import SimpleUNet


# -----------------------------
# Regression U-Net (for conditional denoising)
# -----------------------------
class RegressUNet(nn.Module):
    def __init__(self, target_ch=1, img_ch=3, base=64, tdim=256):
        """
        A U-Net that predicts a regression target (e.g., clean signal or map)
        conditioned on an input image.

        Args:
            target_ch: Number of output channels (e.g., 1 for grayscale target)
            img_ch: Number of conditioning image channels
            base: Base number of feature maps
            tdim: Dimensionality of time embedding
        """
        super().__init__()
        self.net = SimpleUNet(
            in_ch=img_ch + target_ch,   # Concatenate noisy target and conditioning image
            out_ch=target_ch,
            base=base,
            tdim=tdim,
            cond_dim=0
        )

    def forward(self, y_noisy, t, x_img):
        """
        Args:
            y_noisy: Noisy regression target (B, target_ch, H, W)
            t: Diffusion timestep tensor (B,)
            x_img: Conditioning image (B, img_ch, H, W)
        """
        # Concatenate along channel dimension: [y_noisy | x_img]
        x = torch.cat([y_noisy, x_img], dim=1)
        return self.net(x, t, None)
