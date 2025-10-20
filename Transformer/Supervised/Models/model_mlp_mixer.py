import torch
import torch.nn as nn

class MixerBlock(nn.Module):
    def __init__(self, tokens, dim, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.mlp_tokens = nn.Sequential(
            nn.Linear(tokens, int(tokens*mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(tokens*mlp_ratio), tokens), nn.Dropout(dropout)
        )
        self.ln2 = nn.LayerNorm(dim)
        self.mlp_channels = nn.Sequential(
            nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(int(dim*mlp_ratio), dim), nn.Dropout(dropout)
        )
    def forward(self, x):
        # x: (B, N, D)
        x = x + self.mlp_tokens(self.ln1(x).transpose(1,2)).transpose(1,2)
        x = x + self.mlp_channels(self.ln2(x))
        return x

class MLPMixer(nn.Module):
    def __init__(self, num_classes=10, img=32, patch=4, dim=256, depth=6, mlp_ratio=4.0):
        super().__init__()
        self.patch = nn.Conv2d(3, dim, patch, patch)
        self.tokens = (img//patch)*(img//patch)
        self.blocks = nn.ModuleList([MixerBlock(self.tokens, dim, mlp_ratio) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)
    def forward(self, x):
        x = self.patch(x).flatten(2).transpose(1,2)
        for b in self.blocks: x = b(x)
        x = self.norm(x.mean(dim=1))
        return self.head(x)
