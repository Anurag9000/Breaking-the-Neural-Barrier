import torch
import torch.nn as nn
from models.unet_blocks import SimpleUNet


# -----------------------------
# Class-Conditional UNet
# -----------------------------
class ClassCondUNet(nn.Module):
    def __init__(self, num_classes, img_ch=3, base=64, tdim=256):
        super().__init__()
        self.num_classes = num_classes

        # Underlying UNet with conditional embedding
        self.net = SimpleUNet(
            in_ch=img_ch,
            out_ch=img_ch,
            base=base,
            tdim=tdim,
            cond_dim=num_classes
        )

    def forward(self, x, t, y_onehot):
        """
        Args:
            x: input image tensor (B, C, H, W)
            t: timestep tensor (B,)
            y_onehot: one-hot class embedding (B, num_classes)
        """
        return self.net(x, t, y_onehot)

    def forward_sample(self, x, t, y_onehot):
        """
        Args:
            x: input image tensor (B, C, H, W)
            t: timestep tensor (B,)
            y_onehot: one-hot class embedding (B, num_classes)
        """
        return self.net(x, t, y_onehot)

    def forward_predict(self, x, t, y_onehot):
        """
        Args:
            x: input image tensor (B, C, H, W)
            t: timestep tensor (B,)
            y_onehot: one-hot class embedding (B, num_classes)
        """
        return self.net(x, t, y_onehot)
    