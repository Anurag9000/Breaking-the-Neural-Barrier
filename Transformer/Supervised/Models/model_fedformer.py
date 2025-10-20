import torch
import torch.nn as nn
import torch.fft as fft

class FourierBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.lin = nn.Linear(d_model, d_model)
    def forward(self, x):
        # x: (B,L,D)
        X = fft.rfft(x, dim=1)
        X = X * (1 + 0j)  # identity-like; placeholder for spectral weighting
        x_rec = fft.irfft(X, n=x.size(1), dim=1)
        return self.lin(x_rec)

class FEDformer(nn.Module):
    """Simplified FEDformer: Fourier encoder-decoder mixing blocks for forecasting."""
    def __init__(self, d_model=256, e_layers=2, d_layers=1, pred_len=24):
        super().__init__()
        self.enc_in = nn.Linear(1, d_model)
        self.encoder = nn.ModuleList([FourierBlock(d_model) for _ in range(e_layers)])
        self.dec_in = nn.Linear(1, d_model)
        self.decoder = nn.ModuleList([FourierBlock(d_model) for _ in range(d_layers)])
        self.head = nn.Linear(d_model, 1)
        self.pred_len = pred_len
    def forward(self, x_enc, x_dec):
        h = self.enc_in(x_enc)
        for l in self.encoder: h = h + l(h)
        y = self.dec_in(x_dec)
        for l in self.decoder: y = y + l(y)
        return self.head(y)
