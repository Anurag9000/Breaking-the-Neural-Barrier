import torch
import torch.nn as nn
from typing import Tuple


class DAERNNSeq(nn.Module):
    """
    Temporal denoising autoencoder based on stacked GRU/LSTM.

    - Input: sequences (B, C, L). Channels are treated as features.
    - Width controls hidden size; depth controls number of recurrent layers.
    - Uses a bidirectional encoder and a decoder GRU to reconstruct the
      original sequence.
    """

    def __init__(
        self,
        in_channels: int = 1,
        width: int = 64,
        depth: int = 2,
        rnn_type: str = "gru",
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.width = width
        self.depth = depth
        self.rnn_type = rnn_type.lower()

        rnn_cls = nn.GRU if self.rnn_type == "gru" else nn.LSTM

        # Encoder: bidirectional for richer temporal context
        self.encoder = rnn_cls(
            input_size=in_channels,
            hidden_size=width,
            num_layers=depth,
            batch_first=True,
            bidirectional=True,
        )

        # Decoder: unidirectional GRU/LSTM from encoded representation
        self.decoder = rnn_cls(
            input_size=2 * width,
            hidden_size=width,
            num_layers=depth,
            batch_first=True,
        )

        self.output_proj = nn.Linear(width, in_channels)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B,C,L) -> (B,L,C)
        x_seq = x.transpose(1, 2)
        h, _ = self.encoder(x_seq)
        return h

    def decode(self, h: torch.Tensor) -> torch.Tensor:
        z, _ = self.decoder(h)
        out = self.output_proj(z)
        # (B,L,C) -> (B,C,L)
        return out.transpose(1, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        h = self.encode(x)
        x_rec = self.decode(h)
        return x_rec, h


def rnn_total_neurons(width: int, depth: int) -> int:
    return int(width * (depth + 1))

