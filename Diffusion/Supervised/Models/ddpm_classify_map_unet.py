import torch.nn as nn
from models.unet_blocks import SimpleUNet


# -----------------------------
# Classification Map U-Net
# -----------------------------
class ClassifyMapUNet(SimpleUNet):
    """
    A U-Net variant for per-pixel classification diffusion modeling.

    This network predicts class logits (num_classes channels) for each pixel.
    Useful for diffusion models that operate on semantic segmentation maps
    or pixel-wise classification logits.

    Args:
        num_classes: Number of output classes.
        img_ch: Number of input image channels.
        base: Base channel width for the U-Net.
        tdim: Dimensionality of the time embedding.
    """
    def __init__(self, num_classes=10, img_ch=3, base=64, tdim=256):
        super().__init__(
            in_ch=img_ch,
            out_ch=num_classes,
            base=base,
            tdim=tdim,
            cond_dim=0
        )

    def forward(self, logits_noisy, t, x_img):
        """
        Args:
            logits_noisy: Noisy class logits (ignored here, diffusion happens on logits)
            t: Diffusion timestep tensor (B,)
            x_img: Input image tensor (B, img_ch, H, W)

        Note:
            If doing image-level classification, pool logits outside the loss.
            Here we assume per-pixel (segmentation-like) logits diffusion.
        """
        return super().forward(x_img, t, None)
