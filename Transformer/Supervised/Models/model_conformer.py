import torch
import torch.nn as nn

class ConvModule(nn.Module):
    def __init__(self, d_model, expansion=2, kernel=15, dropout=0.1):
        super().__init__()
        self.pw1 = nn.Conv1d(d_model, d_model*expansion, 1)
        self.act = nn.GLU(dim=1)
        self.dw = nn.Conv1d(d_model, d_model, kernel, padding=kernel//2, groups=d_model)
        self.bn = nn.BatchNorm1d(d_model)
        self.swish = nn.SiLU()
        self.pw2 = nn.Conv1d(d_model, d_model, 1)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        # x: (B, T, D)
        x = x.transpose(1, 2)
        x = self.pw1(x)
        x = self.act(x)
        x = self.dw(x)
        x = self.bn(x)
        x = self.swish(x)
        x = self.pw2(x)
        x = x.transpose(1, 2)
        return self.drop(x)

class FeedForwardModule(nn.Module):
    def __init__(self, d_model, expansion=4, dropout=0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, expansion*d_model), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(expansion*d_model, d_model), nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class MHSA(nn.Module):
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        x_ln = self.ln(x)
        out, _ = self.attn(x_ln, x_ln, x_ln, need_weights=False)
        return self.drop(out)

class ConformerBlock(nn.Module):
    def __init__(self, d_model, nhead, conv_kernel=15, ff_expansion=4, dropout=0.1):
        super().__init__()
        self.ff1 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.mhsa = MHSA(d_model, nhead, dropout)
        self.conv = ConvModule(d_model, 2, conv_kernel, dropout)
        self.ff2 = FeedForwardModule(d_model, ff_expansion, dropout)
        self.ln = nn.LayerNorm(d_model)
    def forward(self, x):
        x = x + 0.5 * self.ff1(x)
        x = x + self.mhsa(x)
        x = x + self.conv(x)
        x = x + 0.5 * self.ff2(x)
        return self.ln(x)

class ConformerEncoder(nn.Module):
    def __init__(self, in_feats=80, num_classes=35, d_model=256, nhead=4, layers=6):
        super().__init__()
        self.prenet = nn.Sequential(nn.Linear(in_feats, d_model), nn.ReLU(), nn.Dropout(0.1))
        self.blocks = nn.ModuleList([ConformerBlock(d_model, nhead) for _ in range(layers)])
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.head = nn.Linear(d_model, num_classes)
    def forward(self, feats):
        # feats: (B, T, F)
        x = self.prenet(feats)
        for b in self.blocks:
            x = b(x)
        x = x.transpose(1,2)
        x = self.pool(x).squeeze(-1)
        return self.head(x)
