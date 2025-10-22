import torch
import torch.nn as nn
from models.unet_blocks import SimpleUNet


# -----------------------------
# Pose Estimation U-Net
# -----------------------------
class PoseUNet(nn.Module):
    """
    A U-Net for diffusion-based human pose estimation.
    It predicts clean joint heatmaps given noisy heatmaps and an image.

    Args:
        joints: Number of joints (output heatmap channels)
        img_ch: Number of input image channels
        base: Base feature width for the U-Net
        tdim: Dimensionality of the time embedding
    """
    def __init__(self, joints=17, img_ch=3, base=64, tdim=256):
        super().__init__()
        self.joints = joints
        self.net = SimpleUNet(
            in_ch=img_ch + joints,  # Concatenate heatmap and image
            out_ch=joints,
            base=base,
            tdim=tdim,
            cond_dim=0
        )

    def forward(self, heatmap_noisy, t, img):
        """
        Args:
            heatmap_noisy: Noisy joint heatmap tensor (B, joints, H, W)
            t: Diffusion timestep tensor (B,)
            img: Input image tensor (B, img_ch, H, W)
        """
        x = torch.cat([heatmap_noisy, img], dim=1)
        return self.net(x, t, None)
