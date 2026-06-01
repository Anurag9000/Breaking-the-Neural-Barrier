import torch
import torch.nn as nn
from typing import Tuple

from Diffusion.Supervised.Models.ddpm_audio_spec_unet import AudioSpecUNet
from TRANSFORMERS.Transformer.Supervised.Models.model_convtransformer_ctc import ConvTransformerCTC


class SupDAESpeechSpec(nn.Module):
    """
    Speech spectrogram DAE + ASR head.

    - Denoiser: AudioSpecUNet (spectrogram -> spectrogram).
    - ASR encoder/head: ConvTransformerCTC (spectrogram -> per-frame logits).
    """

    def __init__(self, vocab_size: int = 30, base: int = 64, depth: int = 4):
        super().__init__()
        self.dae = AudioSpecUNet(spec_ch=1, base=base, tdim=256)
        self.asr = ConvTransformerCTC(vocab=vocab_size, d_model=256, depth=depth)
        self.width = base
        self.depth = depth
        self.vocab_size = vocab_size

    def forward(self, spec: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        B, _, H, W = spec.shape
        t = torch.zeros(B, dtype=torch.long, device=spec.device)
        spec_rec = self.dae(spec, t, None)
        logits = self.asr(spec_rec)
        return spec_rec, logits


def sup_dae_speech_total_neurons(width: int, depth: int, vocab_size: int) -> int:
    return int(width * (depth + 1) + vocab_size * width)
