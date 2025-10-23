import torch
import torch.nn as nn
from models.latent_core import DeterministicAutoencoder
from models.lsd_unet import LatentUNet


# -----------------------------------------------------
# Hybrid Diffusion Autoencoder
# Combines a deterministic autoencoder (AE)
# with a latent-space diffusion denoiser (U-Net)
# -----------------------------------------------------
class HybridDiffAE(nn.Module):
    def __init__(self, img_ch=3, z_ch=4):
        super().__init__()

        # Image encoder-decoder (autoencoder)
        self.ae = DeterministicAutoencoder(img_ch=img_ch, z_ch=z_ch)

        # Latent-space diffusion denoiser
        self.denoiser = LatentUNet(z_ch=z_ch)

    def forward(self, x_t_latent, t, _):
        """
        Forward diffusion step:
        - x_t_latent: noisy latent input
        - t: diffusion timestep
        - _: placeholder (e.g., conditioning, unused)
        """
        return self.denoiser(x_t_latent, t, None)
