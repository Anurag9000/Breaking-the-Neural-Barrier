import torch
import torch.nn as nn
from model_swin import Swin, SwinBlock

# ------------------------------
# Swin v2 (lightweight adaptation)
# - We reuse the Swin scaffolding but expose a class to mirror v2 usage.
# - Numerical tweaks (scaled cosine attn etc.) are omitted for brevity; single-model supervised.
# ------------------------------

class SwinV2(Swin):
    def __init__(self, img_size=224, num_classes=10, embed_dim=96, depths=(2,2,6,2), heads=(3,6,12,24), win=8):
        super().__init__(img_size, num_classes, embed_dim, depths, heads, win)
        # could insert minor init tweaks here if desired
