import torch
import torch.nn as nn

class RMSNorm(nn.Module):
    def __init__(self, d):
        super().__init__(); self.w = nn.Parameter(torch.ones(d)); self.eps=1e-6
    def forward(self, x):
        return x * (self.w * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps))

class T5Style(nn.Module):
    """A lightweight T5-style encoder-decoder: RMSNorm, no biases in linear layers, relu activation (~T5 uses GELU variants), shared embeddings optional.
    Note: This is a clean single-model approximation compatible with supervised text-to-text finetunes.
    """
    def __init__(self, src_vocab, tgt_vocab, d_model=256, nhead=8, enc_layers=6, dec_layers=6, ff=1024, dropout=0.1, share_embeddings=False, max_len=1024):
        super().__init__()
        self.src_tok = nn.Embedding(src_vocab, d_model)
        self.tgt_tok = self.src_tok if share_embeddings and (src_vocab==tgt_vocab) else nn.Embedding(tgt_vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.norm = RMSNorm(d_model)
        self.head = nn.Linear(d_model, tgt_vocab, bias=False)

    def _causal(self, T, device):
        m = torch.full((T,T), float('-inf'), device=device); m = torch.triu(m,1); return m

    def forward(self, src, src_pad, tgt_in, tgt_pad):
        B, S = src.shape; T = tgt_in.size(1)
        x = self.src_tok(src) + self.pos(torch.arange(S, device=src.device))
        mem = self.encoder(x, src_key_padding_mask=src_pad)
        y = self.tgt_tok(tgt_in) + self.pos(torch.arange(T, device=src.device))
        out = self.decoder(y, mem, tgt_mask=self._causal(T, y.device), tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        out = self.norm(out)
        return self.head(out)
