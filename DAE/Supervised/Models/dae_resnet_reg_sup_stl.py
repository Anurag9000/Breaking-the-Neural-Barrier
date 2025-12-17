import torch
import torch.nn as nn
from typing import Tuple

from CNN.ADP_ResNet.adp_resnet_backbone import ADPResNet, ADPResNetConfig, estimate_neurons


class SupDAEResNet(nn.Module):
    """
    ResNet backbone with auxiliary image reconstruction head (DAE regulariser).

    - Backbone: ADPResNet classifier.
    - Head: small conv decoder mapping final feature map back to RGB image.
    """

    def __init__(self, num_classes: int = 10, width: int = 16, depth: int = 2):
        super().__init__()
        cfg = ADPResNetConfig(num_classes=num_classes, width=width, depth=depth)
        self.backbone = ADPResNet(cfg)
        self.num_classes = num_classes
        self.width = width
        self.depth = depth

        # Decoder: simple 3x3 conv stack from penultimate feature map to RGB.
        c = self.backbone.final_channels
        self.decoder = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(c, 3, kernel_size=3, padding=1),
        )

    @property
    def input_channels(self) -> int:
        return self.backbone.input_channels

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits, feat = self.backbone.forward_with_features(x)
        # feat: (B,C,H,W) before global pooling.
        x_rec = self.decoder(feat)
        return x_rec, logits


def sup_dae_resnet_total_neurons(width: int, depth: int, num_classes: int) -> int:
    return estimate_neurons(width, depth, num_classes)

