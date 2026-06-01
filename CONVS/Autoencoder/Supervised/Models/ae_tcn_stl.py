import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_TCN_STL: Treat image rows as a sequence (length=H, features=C*W).
# Encoder: 1D dilated causal conv (TCN) over rows; Decoder: mirror with
# ConvTranspose1d. Reshape back to (B,C,H,W). Single-model end-to-end.
# -----------------------------------------------------------------------------

class TCNBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.conv = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class TCNDeBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dilation: int = 1):
        super().__init__()
        pad = dilation
        self.deconv = nn.ConvTranspose1d(in_ch, out_ch, kernel_size=3, padding=pad, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm1d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))

class AE_TCN_STL(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 128, depth: int = 4,
                 dilations: List[int] | int = 1):
        super().__init__()
        assert depth >= 1
        if isinstance(dilations, int):
            self.dilations = [int(dilations)] * depth
        else:
            assert len(dilations) == depth
            self.dilations = [int(d) for d in dilations]
        self.in_channels = in_channels
        self.width = width
        self.depth = depth

        # Flatten per-row: (B,C,H,W) -> (B, H, C*W), then permute to (B, C*W, H)
        self.enc = nn.ModuleList([])
        ch_in = in_channels * 32  # W fixed at 32 for CIFAR
        for i in range(depth):
            d = self.dilations[i]
            self.enc.append(TCNBlock(ch_in, width, dilation=d))
            ch_in = width

        self.dec = nn.ModuleList([])
        for i in reversed(range(depth)):
            d = self.dilations[i]
            ch_out = width if i>0 else in_channels * 32
            if i>0:
                self.dec.append(TCNDeBlock(ch_in, ch_out, dilation=d))
            else:
                self.dec.append(nn.ConvTranspose1d(ch_in, ch_out, kernel_size=3, padding=d, dilation=d))
            ch_in = ch_out

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, (nn.BatchNorm1d)):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        B,C,H,W = x.shape
        h = x.reshape(B, C, H, W).permute(0,2,1,3).reshape(B, H, C*W).permute(0,2,1)  # (B,CW,H)
        for m in self.enc:
            h = m(h)
        return h  # (B,width,H)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = z
        for i, m in enumerate(self.dec):
            h = m(h)
        B, CW, H = h.shape
        C = 3; W = CW // C
        x_rec = h.permute(0,2,1).reshape(B, H, C, W).permute(0,2,1,3)
        return x_rec

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


def ae_tcn_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
