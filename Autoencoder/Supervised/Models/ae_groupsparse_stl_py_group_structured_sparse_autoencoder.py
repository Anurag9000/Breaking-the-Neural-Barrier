import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_GROUPSPARSE_STL: Convolutional AE with group-lasso (channel-wise) sparsity
# on the encoder output. The penalty encourages entire channels to be zeroed by
# summing L2 norms over spatial dims and then L1 over channels.
# The runner computes and adds the group sparsity penalty.
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(out_ch)
        self.a = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.a(self.b(self.c(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.c = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.b = nn.BatchNorm2d(out_ch)
        self.a = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.a(self.b(self.c(x)))

class AE_GROUPSPARSE_STL(nn.Module):
    """
    Group-sparse convolutional autoencoder.

    Args:
        in_channels: input channels
        width: channels per block
        depth: number of encoder blocks
        pool_after: 1-based indices for MaxPool(2) (mirrored in decoder)
    """
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])

        # ---------------- Encoder ----------------
        enc = []
        ch_in = in_channels
        for i in range(1, depth + 1):
            enc.append(ConvBlock(ch_in, width))
            if i in self.pool_after:
                enc.append(nn.MaxPool2d(2, 2))
            ch_in = width
        self.encoder = nn.Sequential(*enc)

        # ---------------- Decoder ----------------
        dec = []
        ch_in = width
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec.append(nn.ConvTranspose2d(ch_in, ch_in, 2, stride=2))
            ch_out = width if i > 1 else in_channels
            if i > 1:
                dec.append(DeconvBlock(ch_in, ch_out))
            else:
                dec.append(nn.ConvTranspose2d(ch_in, ch_out, 3, padding=1))
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec)

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

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


def group_lasso_channels(z: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Channel-wise group lasso on feature map z (B,C,H,W).
    For each sample: sum over channels of sqrt(sum_{h,w} z^2 + eps).
    Returns batch mean.
    """
    # (B,C,H,W) -> (B,C)
    per_channel_l2 = torch.sqrt((z ** 2).sum(dim=(2, 3)) + eps)
    per_sample = per_channel_l2.sum(dim=1)
    return per_sample.mean()


def ae_groupsparse_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
