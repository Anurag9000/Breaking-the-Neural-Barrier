import torch
import torch.nn as nn
from core.diffusion_core import SinusoidalTimeEmbedding


# -----------------------------------------------------
# Patch embedding for latent features
# -----------------------------------------------------
class Patchify(nn.Module):
    def __init__(self, z_ch=4, patch=2, dim=256):
        super().__init__()
        self.proj = nn.Conv2d(z_ch, dim, patch, stride=patch)
        self.patch = patch

    def forward(self, z):
        x = self.proj(z)
        b, c, h, w = x.shape
        # Flatten spatial dimensions into patch tokens
        return x.flatten(2).transpose(1, 2), (h, w)


# -----------------------------------------------------
# DiT-style Transformer for latent diffusion
# -----------------------------------------------------
class DiTLatent(nn.Module):
    def __init__(self, z_ch=4, depth=8, heads=8, dim=256, patch=2):
        super().__init__()

        # Positional encoding (maximum 1024 tokens for now)
        self.pe = nn.Parameter(torch.randn(1, 1024, dim))

        # Time embedding
        self.time = SinusoidalTimeEmbedding(dim)

        # Patch embedding
        self.embed = Patchify(z_ch, patch, dim)

        # Transformer encoder
        enc_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            batch_first=True
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=depth)

        # Projection + unpatchification
        self.proj = nn.Linear(dim, dim)
        self.unpatch = nn.ConvTranspose2d(
            dim, z_ch, kernel_size=patch, stride=patch
        )

    def forward(self, zt, t, _):
        # Embed patches
        tok, (h, w) = self.embed(zt)
        n = tok.size(1)

        # Add time and positional embeddings
        tok = tok + self.time(t).unsqueeze(1) + self.pe[:, :n]

        # Transformer encoding
        tok = self.enc(tok)

        # Project and reshape back to feature map
        tok = self.proj(tok)
        b = tok.shape[0]
        tok = tok.transpose(1, 2).reshape(b, -1, h, w)

        # Reconstruct latent representation
        return self.unpatch(tok)
