import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_MULTI_STL: Multi-scale / Pyramid Autoencoder
# - Encoder: Conv-BN-ReLU blocks with optional MaxPool(2) after selected blocks.
# - Decoder: Mirrors encoder with ConvTranspose2d upsampling where needed.
# - Heads: At *each* decoder stage, emit a reconstruction head (1x1 conv) so we
#          get predictions at multiple spatial scales. Forward returns a list of
#          reconstructions ordered from coarse -> fine (last is full resolution).
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))

class AE_MULTI_STL(nn.Module):
    """
    Multi-scale autoencoder that produces reconstructions at each decoder depth.

    Args:
        in_channels: input channels (3 for RGB)
        width: constant channels for encoder/decoder blocks
        depth: number of Conv blocks in encoder (decoder mirrors)
        pool_after: 1-based indices where a 2x2 MaxPool follows the block
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])

        # ---------------- Encoder ----------------
        enc_blocks: List[nn.Module] = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            ch_out = width
            enc_blocks.append(ConvBlock(ch_in, ch_out))
            if i in self.pool_after:
                enc_blocks.append(nn.MaxPool2d(2, 2))
            ch_in = ch_out
        self.encoder = nn.Sequential(*enc_blocks)

        # ---------------- Decoder + Heads ----------------
        # For each decoder stage i=depth..1, optionally upsample (mirror pool),
        # run a deconv block, then a 1x1 conv head to predict at that scale.
        dec_ops: List[nn.Module] = []
        heads: List[nn.Module] = []
        ch_in = width
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec_ops.append(nn.ConvTranspose2d(ch_in, ch_in, kernel_size=2, stride=2))
            ch_out = width if i > 1 else width  # keep features at 'width' until final head converts
            dec_ops.append(DeconvBlock(ch_in, ch_out))
            heads.append(nn.Conv2d(ch_out, in_channels, kernel_size=1))
            ch_in = ch_out
        self.decoder = nn.ModuleList(dec_ops)  # executed sequentially
        self.heads = nn.ModuleList(list(reversed(heads)))  # align index with stage counting upward
        # Note: We reverse heads so that decoder step 0 uses heads[0], etc.

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

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def decode_multi(self, z: torch.Tensor) -> List[torch.Tensor]:
        """Run decoder and emit a list of reconstructions from coarse->fine.
        We iterate through decoder ops and invoke a prediction head after each
        deconv stage. Returns [x_rec_coarse, ..., x_rec_fine].
        """
        xs = []
        h = z
        head_idx = 0
        for op in self.decoder:
            h = op(h)
            if isinstance(op, DeconvBlock):
                x_rec = self.heads[head_idx](h)
                xs.append(x_rec)
                head_idx += 1
        return xs  # length == depth

    def forward(self, x: torch.Tensor) -> Tuple[List[torch.Tensor], torch.Tensor]:
        z = self.encode(x)
        xs = self.decode_multi(z)
        return xs, z


def ae_multi_total_neurons(width: int, depth: int) -> int:
    # Same scalar proxy as others for capacity plotting
    return int(width * (depth + 1))
