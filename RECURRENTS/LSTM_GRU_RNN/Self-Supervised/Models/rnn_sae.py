import torch
import torch.nn as nn
from typing import Tuple

class GRUSequenceAutoencoder(nn.Module):
    """
    Sequence Autoencoder with GRU encoder and decoder.
    Self-supervised objective: reconstruct the input sequence.
    - Inputs: (T, B, D) or (B, T, D). We support (B, T, D).
    - Teacher forcing during training handled in the runner (feeding target shifted by 1).
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.encoder_gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.enc_to_latent = nn.Linear(hidden_dim * self.num_directions, latent_dim)

        self.latent_to_dec = nn.Linear(latent_dim, hidden_dim * self.num_directions)
        self.decoder_gru = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.proj = nn.Linear(hidden_dim * self.num_directions, input_dim)

    def encode(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # x: (B, T, D)
        enc_out, h_n = self.encoder_gru(x)
        # take last layer hidden state(s): shape (num_layers * directions, B, H)
        last = h_n[-self.num_directions:]  # (directions, B, H)
        last = last.transpose(0,1).reshape(x.size(0), -1)  # (B, H*dir)
        z = self.enc_to_latent(last)  # (B, latent)
        return z, h_n

    def decode(self, z: torch.Tensor, dec_inp: torch.Tensor) -> torch.Tensor:
        # z: (B, latent), dec_inp: (B, T, D)
        dec_h0_flat = self.latent_to_dec(z)  # (B, H*dir)
        # build initial hidden state for all layers; zeros for other layers
        h0 = dec_h0_flat.view(z.size(0), self.num_directions, self.hidden_dim)
        h0 = h0.transpose(0,1)  # (dir, B, H)
        # expand to num_layers
        if self.num_layers > 1:
            pad_layers = torch.zeros(self.num_layers - 1, z.size(0), self.hidden_dim, device=z.device)
            h0 = torch.cat([h0, pad_layers], dim=0)  # (dir + (layers-1), B, H)
        out, _ = self.decoder_gru(dec_inp, h0)
        return self.proj(out)

    def forward(self, x: torch.Tensor, decoder_input: torch.Tensor) -> torch.Tensor:
        z, _ = self.encode(x)
        return self.decode(z, decoder_input)
