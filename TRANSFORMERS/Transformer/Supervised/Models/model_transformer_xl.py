import torch
import torch.nn as nn

class RelPosBias(nn.Module):
    def __init__(self, nhead, max_len=2048):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(nhead, 2*max_len-1))
        self.max_len = max_len
    def forward(self, L_q, L_k):
        # center at max_len-1
        idx = torch.arange(L_q)[:,None] - torch.arange(L_k)[None,:] + (self.max_len-1)
        idx = idx.clamp(0, 2*self.max_len-2)
        return self.bias[:, idx]

class MemTransformerLM(nn.Module):
    """Transformer-XL style: decoder with memory (segment-level recurrence). For simplicity we expose a forward that accepts mem and returns new mem.
    Here used for classification with CLS token at sequence start.
    """
    def __init__(self, vocab, num_classes, d_model=256, nhead=8, layers=6, ff=1024, dropout=0.1, max_mem=128, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        self.pos_bias = RelPosBias(nhead, max_len=max_len)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(d_model, nhead, ff, dropout, batch_first=True, norm_first=True) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.cls_head = nn.Linear(d_model, num_classes)
        self.max_mem = max_mem

    def forward(self, tokens, mem=None):
        # tokens: (B, S)
        B, S = tokens.shape
        h = self.emb(tokens)
        mem = torch.zeros(B, 0, h.size(-1), device=h.device) if mem is None else mem
        tgt = torch.cat([mem, h], dim=1)
        for layer in self.layers:
            T = tgt.size(1)
            causal = torch.full((T,T), float('-inf'), device=tgt.device).triu(1)
            tgt = layer(tgt, torch.zeros_like(tgt), tgt_mask=causal)
        new_mem = tgt[:, -self.max_mem:].detach()
        cls = self.norm(tgt[:, 0])
        return self.cls_head(cls), new_mem
