import torch
import torch.nn as nn

class MovingAvg(nn.Module):
    def __init__(self, kernel=25):
        super().__init__(); self.kernel=kernel
    def forward(self, x):
        # x: (B,L,1)
        pad = self.kernel//2
        w = torch.ones(1,1,self.kernel, device=x.device)/self.kernel
        x1 = x.transpose(1,2)
        y = torch.nn.functional.conv1d(torch.nn.functional.pad(x1, (pad,pad), mode='reflect'), w)
        return y.transpose(1,2)

class AutoCorrelationBlock(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.ln = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
    def forward(self, x):
        x = x + self.attn(self.ln(x), self.ln(x), self.ln(x), need_weights=False)[0]
        x = x + self.ff(self.ln(x))
        return x

class Autoformer(nn.Module):
    """Simplified Autoformer: moving average decomposition + transformer-style blocks."""
    def __init__(self, d_model=256, nhead=8, e_layers=2, d_layers=1, pred_len=24):
        super().__init__()
        self.enc_in = nn.Linear(1, d_model)
        self.dec_in = nn.Linear(1, d_model)
        self.trend = MovingAvg(25)
        self.encoder = nn.ModuleList([AutoCorrelationBlock(d_model, nhead) for _ in range(e_layers)])
        self.decoder = nn.ModuleList([AutoCorrelationBlock(d_model, nhead) for _ in range(d_layers)])
        self.head = nn.Linear(d_model, 1)
        self.pred_len = pred_len
    def forward(self, x_enc, x_dec):
        # decompose
        trend_enc = self.trend(x_enc)
        season_enc = x_enc - trend_enc
        h = self.enc_in(season_enc)
        for l in self.encoder: h = l(h)
        trend_dec = self.trend(x_dec)
        season_dec = x_dec - trend_dec
        y = self.dec_in(season_dec)
        for l in self.decoder: y = l(y)
        out = self.head(y) + trend_dec  # add back trend
        return out
