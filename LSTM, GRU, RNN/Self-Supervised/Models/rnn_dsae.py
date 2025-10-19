import torch
import torch.nn as nn
from typing import Tuple

class GRUDenoisingAutoencoder(nn.Module):
    """
    Denoising Sequence Autoencoder with GRU backbone.
    Corrupt input (mask/drop/gaussian) outside the model; model just reconstructs clean sequence.
    """
    def __init__(self, input_dim: int, hidden_dim: int, latent_dim: int, num_layers: int = 1, bidirectional: bool = False):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.num_directions = 2 if bidirectional else 1

        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.to_latent = nn.Linear(hidden_dim * self.num_directions, latent_dim)

        self.from_latent = nn.Linear(latent_dim, hidden_dim * self.num_directions)
        self.decoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, bidirectional=bidirectional, batch_first=True)
        self.proj = nn.Linear(hidden_dim * self.num_directions, input_dim)

    def forward(self, clean: torch.Tensor, noisy: torch.Tensor) -> torch.Tensor:
        # clean, noisy: (B, T, D)
        enc_out, h = self.encoder(noisy)
        last = h[-self.num_directions:]
        last = last.transpose(0,1).reshape(noisy.size(0), -1)
        z = self.to_latent(last)

        h0_flat = self.from_latent(z)
        h0 = h0_flat.view(noisy.size(0), self.num_directions, self.hidden_dim)
        h0 = h0.transpose(0,1)
        if self.num_layers > 1:
            pad = torch.zeros(self.num_layers - 1, noisy.size(0), self.hidden_dim, device=noisy.device)
            h0 = torch.cat([h0, pad], dim=0)

        # teacher forcing: decoder input provided in runner (shifted clean)
        return z, h0

    def decode(self, h0: torch.Tensor, dec_inp: torch.Tensor) -> torch.Tensor:
        out, _ = self.decoder(dec_inp, h0)
        return self.proj(out)

if __name__ == "__main__":
    B, T, D = 2, 10, 4
    model = GRUDenoisingAutoencoder(D, 16, 16)
    clean = torch.randn(B,T,D)
    noisy = clean + 0.1*torch.randn_like(clean)
    z, h0 = model(clean, noisy)
    dec_inp = torch.zeros_like(clean)
    dec_inp[:,1:,:] = clean[:,:-1,:]
    rec = model.decode(h0, dec_inp)
    print(rec.shape)
