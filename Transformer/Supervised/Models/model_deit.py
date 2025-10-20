import torch
import torch.nn as nn

# -------------------------------------------------------
# DeiT (Data-efficient Image Transformers) — model file
# We use the standard ViT architecture (no distillation token).
# DeiT is primarily a training recipe; architecture == ViT.
# This file is a thin wrapper to keep naming distinct in your repo.
# -------------------------------------------------------

from model_vit import VisionTransformer as _ViT

class DeiT(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.backbone = _ViT(**kwargs)

    def forward(self, x):
        return self.backbone(x)

if __name__ == "__main__":
    m = DeiT(img_size=224, patch_size=16, num_classes=10, embed_dim=384, depth=12, num_heads=6)
    y = m(torch.randn(2,3,224,224))
    print(y.shape)
