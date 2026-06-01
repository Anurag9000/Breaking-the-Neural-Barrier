import torch
import torch.nn as nn
import torch.nn.functional as F

class PosEnc(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2)*(-torch.log(torch.tensor(10000.0))/dim))
        pe[:, 0::2] = torch.sin(pos*div)
        pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class PrefixLM(nn.Module):
    """UniLM/PrefixLM: causal decoding with a bidirectional-encoded prefix.
    Implemented by providing a custom attention mask: tokens within prefix attend bidirectionally;
    tokens in generation region attend to prefix and to previous generation tokens only.
    """
    def __init__(self, vocab, dim=256, depth=6, heads=8, mlp_ratio=4.0, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = PosEnc(dim, max_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=int(dim*mlp_ratio), batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)

    def build_mask(self, B, L, prefix_len, device):
        attn = torch.zeros(L, L, device=device)
        # prefix region: full attention
        attn[:prefix_len, :prefix_len] = 0
        # generation region: causal within itself, full to prefix
        gen = torch.triu(torch.ones(L-prefix_len, L-prefix_len, device=device), diagonal=1)
        attn[prefix_len:, prefix_len:] = gen * -1e9
        attn[prefix_len:, :prefix_len] = 0
        return attn  # (L,L) additive mask

    def forward(self, x, prefix_len):
        # x: (B,L)
        h = self.pos(self.emb(x))
        B,L = x.size(0), x.size(1)
        attn_mask = self.build_mask(B, L, prefix_len, x.device)
        for lyr in self.layers:
            h = lyr(h, src_mask=attn_mask)
        h = self.norm(h)
        logits = self.head(h)
        # LM loss only on generation region
        targets = x[:, 1:].contiguous()
        logits = logits[:, :-1, :].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss
