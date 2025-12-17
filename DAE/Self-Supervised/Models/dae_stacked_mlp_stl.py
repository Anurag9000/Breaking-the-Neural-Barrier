import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

from .dae_saltpepper_mlp_stl import dae_total_neurons


class DAEStackedMLP(nn.Module):
    """
    Fully-connected denoising autoencoder intended for stacked (layer-wise)
    pretraining experiments. The encoder is a stack of Linear+ReLU blocks of
    width `width` repeated `depth` times; the decoder mirrors the encoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 32,
        width: int = 1024,
        depth: int = 4,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.img_size = img_size
        self.width = width
        self.depth = depth

        input_dim = in_channels * img_size * img_size
        self.input_dim = input_dim

        enc_layers: List[nn.Module] = []
        enc_layers.append(nn.Linear(input_dim, width))
        enc_layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 1):
            enc_layers.append(nn.Linear(width, width))
            enc_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        for _ in range(depth - 1):
            dec_layers.append(nn.Linear(width, width))
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers.append(nn.Linear(width, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def _enc_linears(self) -> List[nn.Linear]:
        return [m for m in self.encoder if isinstance(m, nn.Linear)]

    def _dec_linears(self) -> List[nn.Linear]:
        return [m for m in self.decoder if isinstance(m, nn.Linear)]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        z_in = x.view(b, self.input_dim)
        return self.encoder(z_in)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        out = self.decoder(z).view(b, self.in_channels, self.img_size, self.img_size)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z

    def forward_activations(self, x_flat: torch.Tensor, upto_layer: int) -> torch.Tensor:
        """
        Helper used by layer-wise pretraining to obtain activations entering the
        `upto_layer`-th Linear block (0-indexed).
        """
        h = x_flat
        enc_linears = self._enc_linears()
        for idx in range(upto_layer):
            h = F.relu(enc_linears[idx](h))
        return h


__all__ = ["DAEStackedMLP", "dae_total_neurons"]

