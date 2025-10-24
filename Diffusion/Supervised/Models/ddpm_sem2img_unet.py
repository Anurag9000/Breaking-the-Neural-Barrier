import torch
import torch.nn as nn
import torch.nn.functional as F
from models.unet_blocks import SimpleUNet


class Sem2ImgUNet(nn.Module):
    def __init__(self, img_ch=3, num_classes=19, base=64, tdim=256):
        super().__init__()
        self.k = num_classes
        # Input: noisy RGB + semantic one-hot channels
        self.net = SimpleUNet(
            in_ch=img_ch + num_classes,
            out_ch=img_ch,
            base=base,
            tdim=tdim,
            cond_dim=0
        )

    def forward(self, x_t, t, sem_onehot):
        x = torch.cat([x_t, sem_onehot], dim=1)
        return self.net(x, t, None)
