import math
from typing import List, Optional

import torch
import torch.nn as nn

# -----------------------------------------------------------------------------
# Blocks (mirrors Autoencoder/Supervised/Models/ae_denoise_stl.py)
# -----------------------------------------------------------------------------


class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: Optional[int] = None, bias: bool = False):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, p: Optional[int] = None, bias: bool = False):
        super().__init__()
        if p is None:
            p = k // 2
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))


# -----------------------------------------------------------------------------
# ADP-enabled supervised DAE (single-model)
# -----------------------------------------------------------------------------


class ADPDAE(nn.Module):
    """
    Denoising autoencoder with adaptive depth/width.
    - Symmetric conv encoder/decoder with optional pooling at selected indices.
    - Methods `append_depth` and `widen_all` let search policies mutate capacity.
    """

    def __init__(
        self,
        in_channels: int = 3,
        widths: List[int] = (64, 64, 64, 64),
        pooling_indices: List[int] = (),
        bias: bool = False,
    ):
        super().__init__()
        assert len(widths) >= 1, "Need at least one block"
        self.in_channels = in_channels
        self.bias = bias
        self.pooling_indices = sorted(set(pooling_indices))

        # Encoder
        enc_blocks = []
        c = in_channels
        for w in widths:
            enc_blocks.append(ConvBlock(c, w, bias=bias))
            c = w
        self.encoder = nn.ModuleList(enc_blocks)
        self._pools_here = [i in self.pooling_indices for i in range(len(widths))]
        self.pool = nn.MaxPool2d(2, 2)

        # Decoder (mirror)
        dec_blocks = []
        c = widths[-1]
        for i in range(len(widths) - 1, -1, -1):
            c_out = widths[i - 1] if i - 1 >= 0 else in_channels
            stride = 2 if self._pools_here[i] else 1
            outpad = 1 if stride == 2 else 0
            dec_blocks.append(nn.ConvTranspose2d(c, c_out, kernel_size=3, stride=stride, padding=1, output_padding=outpad, bias=bias))
            dec_blocks.append(nn.BatchNorm2d(c_out))
            dec_blocks.append(nn.ReLU(inplace=True))
            c = c_out
        self.decoder = nn.Sequential(*dec_blocks)

        # Head
        self.recon_head = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, bias=True)

        self.apply(self._init_weights)

    # -------------------
    # Properties / stats
    # -------------------
    @property
    def widths(self) -> List[int]:
        return [blk.bn.num_features for blk in self.encoder]

    def depth(self) -> int:
        return len(self.encoder)

    def total_neurons(self) -> int:
        enc = sum([blk.bn.num_features for blk in self.encoder])
        # decoder convtranspose layers mirror encoder count; divide seq length by 3 (convT+BN+ReLU)
        dec = sum([m.num_features for m in self.decoder if isinstance(m, nn.BatchNorm2d)])
        return enc + dec

    # -------------------
    # Forward
    # -------------------
    def forward(self, x_noisy: torch.Tensor) -> torch.Tensor:
        h = x_noisy
        feats = []
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            feats.append(h)
            if self._pools_here[i]:
                h = self.pool(h)
        # Decoder: sequential already includes stride>1 when needed
        h = self.decoder(h)
        out = self.recon_head(h)
        return out

    # -------------------
    # Mutations
    # -------------------
    def append_depth(self) -> None:
        """Append one encoder block (same width as last) and mirrored decoder block."""
        last_c = self.encoder[-1].bn.num_features
        new_enc = ConvBlock(last_c, last_c, bias=self.bias)
        self.encoder.append(new_enc)
        self._pools_here.append(False)

        # Decoder: prepend mirror layers (convT + bn + relu)
        new_dec = [
            nn.ConvTranspose2d(last_c, last_c, kernel_size=3, stride=1, padding=1, output_padding=0, bias=self.bias),
            nn.BatchNorm2d(last_c),
            nn.ReLU(inplace=True),
        ]
        self.decoder = nn.Sequential(*new_dec, *self.decoder)

    def widen_all(self, ex_k: int) -> None:
        """Increase channels of every encoder block by ex_k and rebuild decoder+head."""
        if ex_k <= 0:
            return
        prev = self.in_channels
        for enc in self.encoder:
            old_out = enc.bn.num_features
            new_out = old_out + ex_k
            _resize_conv2d_(enc.conv, in_ch=prev, out_ch=new_out)
            _resize_bn2d_(enc.bn, new_out)
            prev = new_out
        enc_widths = [blk.bn.num_features for blk in self.encoder]
        self._rebuild_decoder(enc_widths)
        _resize_conv2d_(self.recon_head, in_ch=self.in_channels, out_ch=self.in_channels)

    def _rebuild_decoder(self, enc_widths: List[int]) -> None:
        old = list(self.decoder.children())
        new_layers = []
        c = enc_widths[-1]
        for i in range(len(enc_widths) - 1, -1, -1):
            c_out = enc_widths[i - 1] if i - 1 >= 0 else self.in_channels
            stride = 2 if self._pools_here[i] else 1
            outpad = 1 if stride == 2 else 0
            deconv = nn.ConvTranspose2d(c, c_out, kernel_size=3, stride=stride, padding=1, output_padding=outpad, bias=self.bias)
            bn = nn.BatchNorm2d(c_out)
            act = nn.ReLU(inplace=True)
            new_layers.extend([deconv, bn, act])
            c = c_out
        # transplant overlapping weights
        for new_m, old_m in zip([m for m in new_layers if isinstance(m, nn.ConvTranspose2d)],
                                [m for m in old if isinstance(m, nn.ConvTranspose2d)]):
            _overlap_copy_(new_m.weight.data, old_m.weight.data)
            if new_m.bias is not None and old_m.bias is not None:
                _overlap_copy_(new_m.bias.data, old_m.bias.data)
        for new_m, old_m in zip([m for m in new_layers if isinstance(m, nn.BatchNorm2d)],
                                [m for m in old if isinstance(m, nn.BatchNorm2d)]):
            _overlap_copy_(new_m.weight.data, old_m.weight.data)
            _overlap_copy_(new_m.bias.data, old_m.bias.data)
            _overlap_copy_(new_m.running_mean, old_m.running_mean)
            _overlap_copy_(new_m.running_var, old_m.running_var)
        self.decoder = nn.Sequential(*new_layers)

    # -------------------
    # Init
    # -------------------
    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if getattr(m, "bias", None) is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)


# -----------------------------------------------------------------------------
# Resize helpers
# -----------------------------------------------------------------------------


def _overlap_copy_(dst: torch.Tensor, src: torch.Tensor) -> None:
    dims = [min(a, b) for a, b in zip(dst.shape, src.shape)]
    slicer = tuple(slice(0, d) for d in dims)
    dst[slicer].copy_(src[slicer])


def _resize_conv2d_(conv: nn.Conv2d, in_ch: int, out_ch: int) -> None:
    old_w = conv.weight.data.clone()
    old_b = conv.bias.data.clone() if conv.bias is not None else None
    k_h, k_w = conv.kernel_size
    device = conv.weight.device
    conv.in_channels = in_ch
    conv.out_channels = out_ch
    conv.weight = nn.Parameter(torch.empty(out_ch, in_ch, k_h, k_w, device=device))
    nn.init.kaiming_normal_(conv.weight, nonlinearity="relu")
    _overlap_copy_(conv.weight.data, old_w)
    if conv.bias is not None:
        conv.bias = nn.Parameter(torch.zeros(out_ch, device=device))
        if old_b is not None:
            _overlap_copy_(conv.bias.data, old_b)


def _resize_bn2d_(bn: nn.BatchNorm2d, out_ch: int) -> None:
    device = bn.weight.device
    old_w = bn.weight.data.clone()
    old_b = bn.bias.data.clone()
    old_rm = bn.running_mean.clone()
    old_rv = bn.running_var.clone()
    bn.num_features = out_ch
    bn.weight = nn.Parameter(torch.ones(out_ch, device=device))
    bn.bias = nn.Parameter(torch.zeros(out_ch, device=device))
    bn.running_mean = torch.zeros(out_ch, device=device)
    bn.running_var = torch.ones(out_ch, device=device)
    _overlap_copy_(bn.weight.data, old_w)
    _overlap_copy_(bn.bias.data, old_b)
    _overlap_copy_(bn.running_mean, old_rm)
    _overlap_copy_(bn.running_var, old_rv)
