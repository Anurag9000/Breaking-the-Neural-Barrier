import torch
import torch.nn as nn

class BARTStyle(nn.Module):
    """BART-like encoder-decoder for supervised seq2seq (denoising not enforced here)."""
    def __init__(self, vocab: int, d_model=256, nhead=8, enc_layers=6, dec_layers=6, ff=1024, dropout=0.1, max_len=1024):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        dec = nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, enc_layers)
        self.decoder = nn.TransformerDecoder(dec, dec_layers)
        self.head = nn.Linear(d_model, vocab)

    def _causal(self, T, device):
        m = torch.full((T,T), float('-inf'), device=device)
        return torch.triu(m, 1)

    def forward(self, src, src_pad, tgt_in, tgt_pad):
        B, S = src.shape; T = tgt_in.size(1)
        x = self.emb(src) + self.pos(torch.arange(S, device=src.device))
        mem = self.encoder(x, src_key_padding_mask=src_pad)
        y = self.emb(tgt_in) + self.pos(torch.arange(T, device=src.device))
        out = self.decoder(y, mem, tgt_mask=self._causal(T, y.device), tgt_key_padding_mask=tgt_pad, memory_key_padding_mask=src_pad)
        return self.head(out)
