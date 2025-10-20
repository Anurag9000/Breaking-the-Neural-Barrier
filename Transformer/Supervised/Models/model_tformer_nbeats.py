import torch
import torch.nn as nn

class NBeatsBlock(nn.Module):
    def __init__(self, d_in=1, width=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_in, width), nn.ReLU(),
            nn.Linear(width, width), nn.ReLU(),
            nn.Linear(width, d_in)
        )
    def forward(self, x):
        return x + self.fc(x)

class TransformerTS(nn.Module):
    def __init__(self, d_model=256, nhead=8, depth=2):
        super().__init__()
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
    def forward(self, x):
        return self.encoder(x)

class TNBeats(nn.Module):
    """Hybrid: token projection -> transformer encoder -> N-BEATS style residual MLP for forecast."""
    def __init__(self, pred_len=24, d_model=128):
        super().__init__()
        self.inp = nn.Linear(1, d_model)
        self.tr = TransformerTS(d_model, 4, 2)
        self.nbeats = nn.Sequential(NBeatsBlock(d_in=d_model, width=256), NBeatsBlock(d_in=d_model, width=256))
        self.head = nn.Linear(d_model, 1)
        self.pred_len = pred_len
    def forward(self, x_enc, x_dec):
        h = self.inp(x_enc)
        h = self.tr(h)
        h = self.nbeats(h)
        y = self.head(h[:, -x_dec.size(1):, :])
        return y
