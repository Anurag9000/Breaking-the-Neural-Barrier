import torch
import torch.nn as nn

class SparseAttentionEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, block=16, num_global=1, dropout=0.1):
        super().__init__()
        self.block = block; self.num_global = num_global
        self.layer = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, dropout, batch_first=True, norm_first=True)
    def forward(self, x):
        # Approximate BigBird by local block attention via attn_mask (random/global skipped for simplicity)
        B, S, D = x.shape
        mask = torch.full((S,S), float('-inf'), device=x.device)
        for i in range(0, S, self.block):
            l=i; r=min(S, i+self.block)
            mask[l:r, l:r] = 0
        # allow first token to be global
        mask[:, :self.num_global] = 0; mask[:self.num_global, :] = 0
        return self.layer(x, attn_mask=mask)

class BigBirdEncoder(nn.Module):
    def __init__(self, vocab, num_classes, d_model=256, nhead=8, layers=6, block=16, num_global=1, max_len=4096, pad_id=0):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model, padding_idx=pad_id)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([SparseAttentionEncoderLayer(d_model, nhead, block, num_global) for _ in range(layers)])
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.pad_id = pad_id
    def forward(self, ids):
        B,S = ids.shape
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x[:,0])
        return self.head(x)
