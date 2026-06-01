import torch
import torch.nn as nn
from typing import List, Tuple

# -----------------------------------------------------------------------------
# AE_ORTHO_STL: Convolutional AE with orthogonality regularization on conv
# kernels. Penalty is ||W W^T - I||_F^2 where W is reshaped to (out_c, in_c*k*k).
# Runner adds this penalty to MSE reconstruction.
# -----------------------------------------------------------------------------

class ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

class DeconvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, 3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.deconv(x)))

class AE_ORTHO_STL(nn.Module):
    def __init__(self, in_channels: int = 3, width: int = 64, depth: int = 4, pool_after: List[int] = None):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.pool_after = set(pool_after or [])

        enc, ch_in = [], in_channels
        self._enc_convs: List[nn.Conv2d] = []
        for i in range(1, depth+1):
            blk = ConvBlock(ch_in, width)
            enc.append(blk)
            if i in self.pool_after:
                enc.append(nn.MaxPool2d(2,2))
            self._enc_convs.append(blk.conv)
            ch_in = width
        self.encoder = nn.Sequential(*enc)

        dec, ch_in = [], width
        self._dec_convs: List[nn.ConvTranspose2d] = []
        for i in range(depth, 0, -1):
            if i in self.pool_after:
                dec.append(nn.ConvTranspose2d(ch_in, ch_in, 2, stride=2))
            ch_out = width if i>1 else in_channels
            if i>1:
                blk = DeconvBlock(ch_in, ch_out)
                dec.append(blk)
                self._dec_convs.append(blk.deconv)
            else:
                final = nn.ConvTranspose2d(ch_in, ch_out, 3, padding=1)
                dec.append(final)
                self._dec_convs.append(final)
            ch_in = ch_out
        self.decoder = nn.Sequential(*dec)

        self.apply(self._init)

    @staticmethod
    def _init(m: nn.Module):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return self.decoder(z)
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x); x_rec = self.decode(z); return x_rec, z


def orthogonality_penalty(convs: List[nn.Module]) -> torch.Tensor:
    """Sum ||W W^T - I||_F^2 over provided conv/deconv modules.
    For Conv2d: W has shape (out_c, in_c, k, k); reshape to (out_c, in_c*k*k).
    For ConvTranspose2d: treat weight as (in_c, out_c, k, k) and reshape to
    (in_c, out_c*k*k) so orthogonality is applied on its first dim too.
    """
    total = x = 0.0
    device = None
    pen = 0.0
    for m in convs:
        w = m.weight
        if isinstance(m, nn.Conv2d):
            W = w.flatten(1)  # (out_c, in_c*k*k)
        else:  # ConvTranspose2d
            W = w.permute(1,0,2,3).contiguous().flatten(1)  # (out_c, in_c*k*k)
        G = W @ W.t()
        I = torch.eye(G.size(0), device=G.device, dtype=G.dtype)
        pen = pen + ((G - I)**2).sum()
    return pen


def ae_ortho_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))
