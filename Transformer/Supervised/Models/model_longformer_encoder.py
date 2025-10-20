import torch
import torch.nn as nn

class LocalSelfAttention(nn.Module):
    """Sliding-window attention via attention mask; simple, single-model stand-in for Longformer."""
    def __init__(self, d_model: int, nhead: int, window: int, dropout: float = 0.1):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True, norm_first=True)
        self.window = window
    def forward(self, x):
        B, S, D = x.shape
        attn_mask = torch.full((S, S), float('-inf'), device=x.device)
        for i in range(S):
            l = max(0, i - self.window)
            r = min(S, i + self.window + 1)
            attn_mask[i, l:r] = 0
        return self.layer(x, attn_mask=attn_mask)

class LongformerEncoder(nn.Module):
    def __init__(self, vocab: int, num_classes: int, d_model: int = 256, nhead: int = 8, layers: int = 6, window: int = 32, pad_id: int = 0, max_len: int = 4096):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([LocalSelfAttention(d_model, nhead, window) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.pad_id = pad_id
    def forward(self, ids):
        B, S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:, 0])  # use first token as CLS (prepend in run)
        return self.head(x)
