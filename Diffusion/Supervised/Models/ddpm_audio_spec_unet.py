import torch.nn as nn
from models.unet_blocks import SimpleUNet


# -----------------------------
# Audio Spectrogram U-Net
# -----------------------------
class AudioSpecUNet(SimpleUNet):
    """
    A U-Net for diffusion-based denoising of audio spectrograms.

    Args:
        spec_ch: Number of spectrogram channels (e.g., 1 for magnitude)
        base: Base feature width of U-Net
        tdim: Dimensionality of the time embedding
    """
    def __init__(self, spec_ch=1, base=64, tdim=256):
        super().__init__(
            in_ch=spec_ch,   # Input spectrogram channels
            out_ch=spec_ch,  # Output spectrogram channels
            base=base,
            tdim=tdim,
            cond_dim=0       # No conditional input
        )

    def forward(self, spec_noisy, t, _cond=None):
        """
        Args:
            spec_noisy: Noisy spectrogram tensor (B, spec_ch, H, W)
            t: Diffusion timestep tensor (B,)
            _cond: Placeholder for conditional input (ignored)
        """
        return super().forward(spec_noisy, t, None)
