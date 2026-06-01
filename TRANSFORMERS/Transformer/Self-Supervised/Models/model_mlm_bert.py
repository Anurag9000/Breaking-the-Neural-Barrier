import torch
import torch.nn as nn
import torch.nn.functional as F

class PosEnc(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2)*(-torch.log(torch.tensor(10000.0))/dim))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class BERTEncoder(nn.Module):
    def __init__(self, vocab, dim=512, depth=6, heads=8, mlp_ratio=4.0, max_len=256):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = PosEnc(dim, max_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=int(dim*mlp_ratio), batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.mlm = nn.Linear(dim, vocab)

    def mask_inputs(self, x, mask_token_id=103, mask_prob=0.15):
        # x: (B,L) integer tokens
        B,L = x.shape
        mask = torch.rand(B, L, device=x.device) < mask_prob
        targets = x.clone()
        x_masked = x.clone()
        x_masked[mask] = mask_token_id
        return x_masked, targets, mask

    def forward(self, x, mask_token_id=103, mask_prob=0.15):
        x_in, targets, mask = self.mask_inputs(x, mask_token_id, mask_prob)
        h = self.pos(self.emb(x_in))
        for lyr in self.layers:
            h = lyr(h)
        h = self.norm(h)
        logits = self.mlm(h)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), reduction='none').view_as(targets)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss
