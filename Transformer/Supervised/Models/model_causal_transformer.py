import torch
import torch.nn as nn
import math

class SinusoidalPE(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]

class CausalTransformer(nn.Module):
    """GPT-style decoder-only Transformer for supervised classification via CLS token."""
    def __init__(self, vocab_size: int, num_classes: int, d_model: int = 256, nhead: int = 8,
                 num_layers: int = 6, dim_ff: int = 1024, dropout: float = 0.1, max_len: int = 1024,
                 pad_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.d_model = d_model
        self.nhead = nhead
        self.num_layers = num_layers
        self.dim_ff = dim_ff
        self.dropout = dropout
        self.max_len = max_len
        self.pad_id = pad_id
        
        self.tok = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        nn.init.normal_(self.tok.weight, mean=0.0, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.normal_(self.cls, mean=0.0, std=0.02)
        self.pos = SinusoidalPE(d_model, max_len + 1)
        dec_layer = nn.TransformerDecoderLayer(d_model, nhead, dim_ff, dropout, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)


    def _causal_mask(self, T, device):
        m = torch.full((T, T), float('-inf'), device=device)
        return torch.triu(m, diagonal=1)

    def forward(self, tokens: torch.Tensor):
        # tokens: (B, S)
        B, S = tokens.size()
        x = self.tok(tokens)
        cls_tok = self.cls.expand(B, -1, -1)
        x = torch.cat([cls_tok, x], dim=1)
        x = self.pos(x)
        T = x.size(1)
        tgt_mask = self._causal_mask(T, x.device)
        # use empty memory (zeros) – decoder-only; implement as self-attention via decoder with None memory
        mem = torch.zeros(B, 1, x.size(-1), device=x.device)
        out = self.decoder(x, mem.expand(B, 1, -1), tgt_mask=tgt_mask)
        cls = self.norm(out[:, 0])
        return self.head(cls)
