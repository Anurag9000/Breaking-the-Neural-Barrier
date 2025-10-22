import torch
import torch.nn as nn
from models.unet_blocks import SimpleUNet


# -----------------------------
# Flow U-Net for Optical Flow Diffusion
# -----------------------------
class FlowUNet(nn.Module):
    """
    A U-Net for diffusion-based denoising of optical flow fields,
    conditioned on an image pair (e.g., I1, I2).

    Args:
        flow_ch: Number of flow channels (typically 2: u,v)
        img_ch: Number of channels per image (e.g., 3 for RGB)
        base: Base feature width of the U-Net
        tdim: Dimensionality of the time embedding
    """
    def __init__(self, flow_ch=2, img_ch=6, base=64, tdim=256):
        super().__init__()
        # Input: noisy flow + concatenated image pair
        self.net = SimpleUNet(
            in_ch=img_ch + flow_ch,  # flow + image pair
            out_ch=flow_ch,          # predict flow channels
            base=base,
            tdim=tdim,
            cond_dim=0               # no extra conditional embedding
        )

    def forward(self, flow_noisy, t, img_pair):
        """
        Args:
            flow_noisy: Noisy flow tensor (B, flow_ch, H, W)
            t: Diffusion timestep tensor (B,)
            img_pair: Concatenated image pair tensor (B, img_ch, H, W)

        Returns:
            Denoised flow tensor (B, flow_ch, H, W)
        """
        x = torch.cat([flow_noisy, img_pair], dim=1)
        return self.net(x, t, None)
