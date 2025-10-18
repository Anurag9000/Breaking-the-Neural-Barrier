import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_TIED_STL: Convolutional autoencoder with tied weights (decoder uses the
# transpose of encoder conv weights via F.conv_transpose2d). Optional pooling in
# encoder is mirrored by stride-2 ConvTranspose2d in decoder (these upsamplers
# are not tied, only the main conv blocks are weight-shared).
# -----------------------------------------------------------------------------

class EncConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DecTiedBlock(nn.Module):
    """
    Decoder block whose weight is *tied* to a given encoder Conv2d weight.
    We implement the deconvolution using F.conv_transpose2d with the encoder's
    conv weight tensor (shape [out_c, in_c, k, k]) which matches the expected
    deconv weight shape ([in_c, out_c, k, k]).
    """
    def __init__(self, enc_conv: nn.Conv2d, out_ch: int, use_bn: bool = True):
        super().__init__()
        self.enc_conv = enc_conv  # reference to encoder conv (for weight sharing)
        self.use_bn = use_bn
        if use_bn:
            self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Note: padding=1 assumes 3x3 kernel; align with encoder.
        w = self.enc_conv.weight
        y = F.conv_transpose2d(x, w, bias=None, stride=1, padding=1)
        if self.use_bn:
            y = self.bn(y)
        return self.act(y)

class AE_TIED_STL(nn.Module):
    """
    Tied-weights Autoencoder.

    Args:
      in_channels: input channels (3 for RGB)
      width: channels per block (constant across blocks)
      depth: number of encoder Conv blocks (decoder mirrors)
      pool_after: 1-based indices where MaxPool2d(2) is inserted after block
                  (decoder mirrors with ConvTranspose2d(stride=2))
      decoder_bn: whether to use BN in decoder tied blocks
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4,
                 pool_after: List[int] = None, decoder_bn: bool = True):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])
        self.decoder_bn = decoder_bn

        # ----------------- Encoder -----------------
        self.enc_blocks: List[nn.Module] = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            block = EncConvBlock(ch_in, width)
            self.enc_blocks.append(block)
            if i in self.pool_after:
                self.enc_blocks.append(nn.MaxPool2d(2, 2))
            ch_in = width
        self.encoder = nn.Sequential(*self.enc_blocks)

        # Keep references to the *Conv2d* modules (exclude pools)
        self._enc_convs: List[nn.Conv2d] = [m.conv for m in self.enc_blocks if isinstance(m, EncConvBlock)]

        # ----------------- Decoder (tied) -----------------
        dec_blocks: List[nn.Module] = []
        # Mirror in reverse: for block index i=depth..1
        conv_idx = len(self._enc_convs) - 1
        ch_in = width
        for i in range(depth, 0, -1):
            # First mirror pooling if present after encoder block i
            if i in self.pool_after:
                dec_blocks.append(nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2))
            # Then a tied deconv block; for the last stage, output is in_channels
            out_ch = width if i > 1 else in_channels
            # Tie to the matching encoder conv: index conv_idx
            tied_block = DecTiedBlock(self._enc_convs[conv_idx], out_ch, use_bn=self.decoder_bn if i > 1 else False)
            conv_idx -= 1
            if i > 1:
                dec_blocks.append(tied_block)
            else:
                # Final layer: produce in_channels, but keep tying via conv_transpose
                # and *no activation* at the end; replace ReLU with identity
                class FinalTied(nn.Module):
                    def __init__(self, base: DecTiedBlock):
                        super().__init__()
                        self.base = base
                    def forward(self, x: torch.Tensor) -> torch.Tensor:
                        w = self.base.enc_conv.weight
                        y = F.conv_transpose2d(x, w, bias=None, stride=1, padding=1)
                        return y  # no BN, no activation
                dec_blocks.append(FinalTied(tied_block))
            ch_in = out_ch
        self.decoder = nn.Sequential(*dec_blocks)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
        # Note: no standalone ConvTranspose2d weights in tied blocks

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


def ae_tied_total_neurons(width: int, depth: int) -> int:
    # Same scalar capacity proxy as others
    return int(width * (depth + 1))
