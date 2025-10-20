import torch
import torch.nn as nn

class ProbSparseAttention(nn.Module):
    """Simplified ProbSparse: keep top-k keys per query based on dot products; here k is fraction of sequence length."""
    def __init__(self, d_model, nhead, k_frac=0.25):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, batch_first=True)
        self.k_frac = k_frac
    def forward(self, q, k, v):
        # naive fallback: full attention (for stability). For k_frac < 1, could implement masking but keep simple here.
        out, _ = self.attn(q, k, v, need_weights=False)
        return out

class EncoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.attn = ProbSparseAttention(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
        self.ln2 = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + self.attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x

class DecoderLayer(nn.Module):
    def __init__(self, d_model, nhead):
        super().__init__()
        self.self_attn = ProbSparseAttention(d_model, nhead)
        self.cross_attn = ProbSparseAttention(d_model, nhead)
        self.ln1 = nn.LayerNorm(d_model); self.ln2 = nn.LayerNorm(d_model); self.ln3 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, 4*d_model), nn.GELU(), nn.Linear(4*d_model, d_model))
    def forward(self, x, mem):
        x = x + self.self_attn(self.ln1(x), self.ln1(x), self.ln1(x))
        x = x + self.cross_attn(self.ln2(x), self.ln2(mem), self.ln2(mem))
        x = x + self.ff(self.ln3(x))
        return x

class Informer(nn.Module):
    """Simplified Informer for forecasting: encoder-decoder with ProbSparse attention stubs."""
    def __init__(self, in_feats=1, d_model=256, nhead=8, e_layers=2, d_layers=1, pred_len=24):
        super().__init__()
        self.enc_in = nn.Linear(in_feats, d_model)
        self.encoder = nn.ModuleList([EncoderLayer(d_model, nhead) for _ in range(e_layers)])
        self.dec_in = nn.Linear(in_feats, d_model)
        self.decoder = nn.ModuleList([DecoderLayer(d_model, nhead) for _ in range(d_layers)])
        self.head = nn.Linear(d_model, 1)
        self.pred_len = pred_len
    def forward(self, x_enc, x_dec):
        # x_enc: (B, L, F) history; x_dec: (B, L_out, F) known decoder inputs (zeros)
        h = self.enc_in(x_enc)
        for l in self.encoder: h = l(h)
        y = self.dec_in(x_dec)
        for l in self.decoder: y = l(y, h)
        return self.head(y)  # (B, L_out, 1)
