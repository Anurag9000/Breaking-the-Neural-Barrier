import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------------------------------------------------------
# Blocks (mirrors Autoencoder/Self-Supervised/Models/ae_denoise.py)
# -----------------------------------------------------------------------------


class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: int = 1, bias: bool = True):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# -----------------------------------------------------------------------------
# ADP-enabled self-supervised DAE (single-model)
# -----------------------------------------------------------------------------


class ADPDenoisingConvAE(nn.Module):
    """
    Convolutional denoising AE with adaptive depth/width and optional pooling.
    Corruptions are external; forward expects a noisy input and reconstructs.
    """

    def __init__(
        self,
        in_ch: int = 3,
        widths: List[int] = (16, 32, 64),
        pooling_indices: List[int] = (0, 2),
        use_transpose_conv: bool = False,
    ):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.widths = list(widths)
        self.pooling_indices = set(pooling_indices)
        self.use_transpose_conv = use_transpose_conv

        # Encoder
        enc = []
        ch = in_ch
        for w in widths:
            enc.append(ConvBNReLU(ch, w))
            ch = w
        self.encoder = nn.ModuleList(enc)

        # Decoder (mirror)
        rev_widths = list(reversed(widths))
        dec = []
        ch = rev_widths[0]
        for w in rev_widths[1:]:
            dec.append(ConvBNReLU(ch, w))
            ch = w
        self.decoder = nn.ModuleList(dec)
        self.head = nn.Conv2d(ch, in_ch, kernel_size=1, stride=1, padding=0)

        self.pool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

    # -------------------
    # Stats
    # -------------------
    def total_neurons(self) -> int:
        enc = sum([m.conv.out_channels for m in self.encoder])
        dec = sum([m.conv.out_channels for m in self.decoder])
        return enc + dec

    def depth(self) -> int:
        return len(self.widths)

    def widths_list(self) -> List[int]:
        return list(self.widths)

    # -------------------
    # Forward
    # -------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encode
        downsamples = []
        h = x
        for i, block in enumerate(self.encoder):
            h = block(h)
            if i in self.pooling_indices:
                h = self.pool(h)
                downsamples.append(i)

        # Decode (upsample pending downs)
        pending = len(downsamples)
        for block in self.decoder:
            if pending > 0:
                h = self.upsample(h)
                pending -= 1
            h = block(h)
        while pending > 0:
            h = self.upsample(h)
            pending -= 1
        out = self.head(h)
        return out

    # -------------------
    # Mutations
    # -------------------
    def append_depth(self) -> None:
        """Add one encoder block (same width as last) and mirrored decoder block."""
        last_c = self.encoder[-1].conv.out_channels
        self.encoder.append(ConvBNReLU(last_c, last_c))
        self.widths.append(last_c)
        self.pooling_indices = set(self.pooling_indices)  # ensure valid after append

        # decoder: prepend mirror
        self.decoder.insert(0, ConvBNReLU(last_c, last_c))

    def widen_all(self, ex_k: int) -> None:
        """Increase channels of every block by ex_k."""
        if ex_k <= 0:
            return
        prev = self.in_ch
        for blk in self.encoder:
            old_out = blk.conv.out_channels
            new_out = old_out + ex_k
            _resize_conv2d_(blk.conv, prev, new_out)
            _resize_bn2d_(blk.bn, new_out)
            prev = new_out
        self.widths = [w + ex_k for w in self.widths]
        self._rebuild_decoder(self.widths)
        _resize_conv2d_(self.head, in_ch=self.head.in_channels, out_ch=self.head.out_channels)

    def _rebuild_decoder(self, enc_widths: List[int]) -> None:
        old_dec = self.decoder
        dec = nn.ModuleList()
        ch = enc_widths[-1]
        for i in range(len(enc_widths) - 1, -1, -1):
            w_out = enc_widths[i - 1] if i - 1 >= 0 else self.in_ch
            dec.append(ConvBNReLU(ch, w_out))
            ch = w_out
        for nb, ob in zip(dec, old_dec):
            _overlap_copy_(nb.conv.weight.data, ob.conv.weight.data)
            _overlap_copy_(nb.bn.weight.data, ob.bn.weight.data)
            _overlap_copy_(nb.bn.bias.data, ob.bn.bias.data)
            _overlap_copy_(nb.bn.running_mean, ob.bn.running_mean)
            _overlap_copy_(nb.bn.running_var, ob.bn.running_var)
        self.decoder = dec


# -----------------------------------------------------------------------------
# Simple corruption helpers (self-supervised)
# -----------------------------------------------------------------------------


def corrupt_gaussian(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(-1.0, 1.0)


def corrupt_pixel_mask(x: torch.Tensor, mask_prob: float) -> torch.Tensor:
    if mask_prob <= 0:
        return x
    B, C, H, W = x.shape
    mask = (torch.rand(B, 1, H, W, device=x.device) < mask_prob).float()
    return x * (1.0 - mask)


def corrupt_patch_mask(x: torch.Tensor, mask_ratio: float, patch_size: int = 4) -> torch.Tensor:
    if mask_ratio <= 0:
        return x
    B, C, H, W = x.shape
    assert H % patch_size == 0 and W % patch_size == 0, "H and W must be divisible by patch_size"
    gh, gw = H // patch_size, W // patch_size
    patch_mask = (torch.rand(B, 1, gh, gw, device=x.device) < mask_ratio).float()
    mask = F.interpolate(patch_mask, size=(H, W), mode="nearest")
    return x * (1.0 - mask)


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
