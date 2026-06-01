import torch
import torch.nn as nn

class GPTTagger(nn.Module):
    def __init__(self, vocab, num_tags, d_model=256, nhead=8, layers=6, ff=1024, dropout=0.1, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.tag_head = nn.Linear(d_model, num_tags)
    def forward(self, ids):
        B,S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        tgt = x
        for layer in self.layers:
            causal = torch.full((S,S), float('-inf'), device=x.device).triu(1)
            tgt = layer(tgt, torch.zeros_like(tgt), tgt_mask=causal)
        z = self.norm(tgt)
        return self.tag_head(z)
