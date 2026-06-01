import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleTokenizer:
    def __init__(self, texts, min_freq=1):
        from collections import Counter
        cnt = Counter()
        for t in texts:
            cnt.update(t.strip().split())
        self.itos = ['<pad>', '<unk>'] + [w for w,c in cnt.items() if c>=min_freq]
        self.stoi = {w:i for i,w in enumerate(self.itos)}
    def encode(self, t, max_len):
        ids = [self.stoi.get(w,1) for w in t.strip().split()][:max_len]
        ids += [0]*(max_len-len(ids))
        return torch.tensor(ids, dtype=torch.long)

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

class TransformerEncoder(nn.Module):
    def __init__(self, vocab, dim=256, depth=4, heads=8, mlp_ratio=4.0, max_len=64):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = PosEnc(dim, max_len)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=int(dim*mlp_ratio), batch_first=True)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        h = self.pos(self.emb(x))
        for lyr in self.layers:
            h = lyr(h)
        h = self.norm(h)
        return h[:,0]  # use first token as sentence rep (CLS-free variant)

class SimCSETransformer(nn.Module):
    def __init__(self, vocab, dim=256, depth=4, heads=8, mlp_ratio=4.0, proj_dim=128):
        super().__init__()
        self.encoder = TransformerEncoder(vocab, dim, depth, heads, mlp_ratio)
        self.proj = nn.Linear(dim, proj_dim)
    def forward(self, x1, x2, temperature=0.05):
        z1 = F.normalize(self.proj(self.encoder(x1)), dim=-1)
        z2 = F.normalize(self.proj(self.encoder(x2)), dim=-1)
        z = torch.cat([z1,z2], dim=0)
        sim = (z @ z.t()) / temperature
        B = z1.size(0)
        mask = torch.eye(2*B, device=z.device).bool()
        sim = sim.masked_fill(mask, -9e15)
        targets = torch.cat([torch.arange(B,2*B), torch.arange(0,B)]).to(z.device)
        loss = F.cross_entropy(sim, targets)
        return loss
