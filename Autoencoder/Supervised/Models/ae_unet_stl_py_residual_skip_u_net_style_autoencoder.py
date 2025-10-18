import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_UNET_STL: U-Net-style autoencoder with encoder-decoder skips (single model).
# - Encoder: Conv-BN-ReLU blocks with optional MaxPool(2) after selected blocks.
# - Decoder: Upsample with ConvTranspose2d where needed, concatenate skip from
#            encoder, then refine with Conv-BN-ReLU blocks.
# - Final head: 3x3 ConvTranspose2d to reconstruct input channels; identity act.
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class RefineBlock(nn.Module):
    """Two-layer refinement after skip concatenation (keeps channels at width)."""
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c1 = ConvBlock(in_ch, out_ch)
        self.c2 = ConvBlock(out_ch, out_ch)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c1(x)
        x = self.c2(x)
        return x

class AE_UNET_STL(nn.Module):
    """
    U-Net-style convolutional autoencoder with skip connections.

    Args:
        in_channels: input channels (3 for RGB)
        width: channels per encoder block
        depth: number of encoder Conv blocks
        pool_after: 1-based indices where 2x2 MaxPool follows the block (mirrored)
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])

        # ---------------- Encoder ----------------
        enc_seq: List[nn.Module] = []
        ch_in = in_channels
        self._enc_blocks: List[ConvBlock] = []
        for i in range(1, depth + 1):
            block = ConvBlock(ch_in, width)
            self._enc_blocks.append(block)
            enc_seq.append(block)
            if i in self.pool_after:
                enc_seq.append(nn.MaxPool2d(2, 2))
            ch_in = width
        self.encoder = nn.Sequential(*enc_seq)

        # ---------------- Decoder ----------------
        dec_ops: List[nn.Module] = []
        refine_ops: List[nn.Module] = []
        ch_in = width
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec_ops.append(nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2))
            # After upsample, we concatenate with encoder feature of channels=width
            # so input to refine block is ch_in + width (typically 2*width)
            in_cat = ch_in + (width if i > 0 else 0)
            refine_ops.append(RefineBlock(in_cat, width if i > 1 else width))
            ch_in = width
        self.dec_ups = nn.ModuleList(dec_ops)
        self.dec_refine = nn.ModuleList(list(reversed(refine_ops)))  # align with forward stage order

        # final head (produce in_channels)
        self.head = nn.ConvTranspose2d(width, in_channels, kernel_size=3, padding=1)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
            if getattr(m, 'bias', None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)

    def _encode_collect(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run encoder while collecting skip tensors *after* each ConvBlock.
        Skips list has length == depth, aligned with block indices 1..depth.
        """
        skips: List[torch.Tensor] = []
        h = x
        block_idx = 0
        for m in self.encoder:
            h = m(h)
            if isinstance(m, ConvBlock):
                skips.append(h)
                block_idx += 1
        return skips

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Encoder with skip capture
        skips = self._encode_collect(x)
        z = skips[-1]  # bottleneck feature (post last ConvBlock, pre pooling if any)

        # Decoder: iterate stages from i=depth..1
        h = z
        up_idx = 0
        refine_idx = 0
        # We must replay encoder modules to apply any pools after the last block
        # but we already captured features at ConvBlock outputs.
        # Build a list of which stages had pooling to align upsamplers
        pool_flags = [ (i in self.pool_after) for i in range(1, self.depth + 1) ]
        for i in range(self.depth, 0, -1):
            if pool_flags[i-1]:
                # apply next upsampler in sequence order
                h = self.dec_ups[up_idx](h)
                up_idx += 1
            # concat skip from encoder stage i-1 (0-based)
            skip = skips[i-1]
            # In case of slight shape mismatch due to rounding, center-crop skip
            if skip.shape[-2:] != h.shape[-2:]:
                dh = min(skip.shape[-2], h.shape[-2])
                dw = min(skip.shape[-1], h.shape[-1])
                skip = skip[..., :dh, :dw]
                h = h[..., :dh, :dw]
            h = torch.cat([h, skip], dim=1)
            h = self.dec_refine[refine_idx](h)
            refine_idx += 1
        x_rec = self.head(h)
        return x_rec, z


def ae_unet_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
