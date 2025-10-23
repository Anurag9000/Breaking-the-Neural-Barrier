import torch
import torch.nn as nn
from core.diffusion_core import SinusoidalTimeEmbedding


# -----------------------------------------------------
# Small feature extractor backbone
# -----------------------------------------------------
class SmallBackbone(nn.Module):
    def __init__(self, in_ch=3, c=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, c, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)


# -----------------------------------------------------
# Feature-map diffusion model
# -----------------------------------------------------
class FMDiffusion(nn.Module):
    def __init__(self, feat_ch=64, tdim=256):
        super().__init__()

        # Time embedding
        self.time = nn.Sequential(
            SinusoidalTimeEmbedding(tdim),
            nn.SiLU()
        )

        # Input and mid layers
        self.inp = nn.Conv2d(feat_ch, feat_ch, 3, padding=1)
        self.mid = nn.Sequential(
            nn.GroupNorm(8, feat_ch),
            nn.SiLU(),
            nn.Conv2d(feat_ch, feat_ch, 3, padding=1)
        )

        # Output layer
        self.out = nn.Conv2d(feat_ch, feat_ch, 3, padding=1)

    def forward(self, ft_noisy, t, _):
        # Embed time and broadcast to spatial dims
        t_emb = self.time(t).unsqueeze(-1).unsqueeze(-1)

        # Combine input features with time embedding
        h = self.inp(ft_noisy) + t_emb

        # Process through mid layers
        h = self.mid(h)

        # Output predicted denoised features
        return self.out(h)
