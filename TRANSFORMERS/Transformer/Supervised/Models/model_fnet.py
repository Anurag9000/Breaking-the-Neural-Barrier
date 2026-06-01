import torch
import torch.nn as nn
import torch.fft as fft

class FNetBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim*mlp_ratio), dim), nn.Dropout(dropout)
        )
    def forward(self, x):
        # token mixing via 2D FFT over (batch, seq) independently per channel
        z = self.ln1(x)
        Z = fft.fftn(z, dim=(1,))  # FFT over sequence dim only
        x = x + Z.real
        x = x + self.mlp(self.ln2(x))
        return x

class FNetClassifier(nn.Module):
    def __init__(self, vocab, num_classes, dim=256, depth=6, max_len=512):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.blocks = nn.ModuleList([FNetBlock(dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    def forward(self, ids):
        x = self.emb(ids)
        for b in self.blocks: x = b(x)
        return self.head(self.norm(x[:,0]))
