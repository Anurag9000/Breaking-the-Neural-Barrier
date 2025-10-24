import torch.nn as nn
from models.unet_blocks import SimpleUNet


class TextCondUNet(nn.Module):
    """U-Net variant that conditions on text embeddings via the time-embedding MLP."""

    def __init__(self, img_ch=3, text_dim=512, base=64, tdim=256):
        super().__init__()
        # Pass the text vector through the time-embedding MLP (via cond_dim)
        self.net = SimpleUNet(
            in_ch=img_ch,
            out_ch=img_ch,
            base=base,
            tdim=tdim,
            cond_dim=text_dim,
        )

    def forward(self, x_t, t, text_vec):
        return self.net(x_t, t, text_vec)
