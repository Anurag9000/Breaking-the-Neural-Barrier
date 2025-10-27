import torch, torch.nn as nn
from ADP_RetNet_model import copy_linear_overlap

# Simple series decomposition with moving-average trend; residual = seasonality
class SeriesDecomp(nn.Module):
    def __init__(self, kernel=5):
        super().__init__()
        pad = kernel//2
        self.avg = nn.Conv1d(1,1,kernel, padding=pad, bias=False)
        with torch.no_grad():
            self.avg.weight[:] = 1.0/kernel
    def forward(self, x):
        # x: B,N,D -> apply per feature over N
        B,N,D = x.shape
        y = x.transpose(1,2).reshape(B*D,1,N)
        trend = self.avg(y)
        trend = trend.view(B,D,N).transpose(1,2)
        season = x - trend
        return trend, season

class AutoCorrBlock(nn.Module):
    def __init__(self, dim, nhead=4, drop=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.nhead = nhead
    def forward(self, x):
        # use FFT-based autocorrelation across tokens -> weights
        h = self.norm(x)
        H = torch.fft.rfft(h, dim=1)
        P = H * torch.conj(H)
        w = torch.fft.irfft(P, n=h.size(1), dim=1).real
        w = torch.softmax(w, dim=1)
        y = torch.bmm(w.transpose(1,2), h) / (w.sum(dim=1, keepdim=True).transpose(1,2)+1e-6)
        return y

class AutoformerBlock(nn.Module):
    def __init__(self, dim, nhead=4, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.decomp = SeriesDecomp(kernel=5)
        self.auto = AutoCorrBlock(dim)
        self.drop = nn.Dropout(drop)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mlp_ratio*dim), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp_ratio*dim, dim), nn.Dropout(drop))
    def forward(self, x):
        trend, season = self.decomp(x)
        y = self.auto(season)
        x = trend + self.drop(y)
        x = x + self.ff(x)
        return x

class AutoformerTiny(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=2):
        super().__init__()
        self.embed_dim=embed_dim
        self.tokenizer = nn.Linear(3, embed_dim)
        self.blocks = nn.ModuleList([AutoformerBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def add_block(self):
        self.blocks.append(AutoformerBlock(self.embed_dim))
    def widen_all(self, ex_k):
        new_dim = self.embed_dim + ex_k
        new_tok = nn.Linear(3, new_dim)
        with torch.no_grad(): copy_linear_overlap(self.tokenizer, new_tok)
        self.tokenizer = new_tok
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = AutoformerBlock(new_dim)
            for (ol, nl) in zip(b.ff, nb.ff):
                if isinstance(ol, nn.Linear) and isinstance(nl, nn.Linear): copy_linear_overlap(ol, nl)
            new_blocks.append(nb)
        self.blocks = new_blocks
        self.norm = nn.LayerNorm(new_dim)
        new_head = nn.Linear(new_dim, self.head.out_features); copy_linear_overlap(self.head, new_head); self.head = new_head
        self.embed_dim = new_dim
    def forward(self, x):
        B,C,H,W = x.shape
        x = x.view(B, C, H*W).transpose(1,2)  # B,N,3
        x = self.tokenizer(x)
        for b in self.blocks: x = b(x)
        x = self.norm(x).mean(1)
        return self.head(x)


def build_autoformer(num_classes=10, init_width=64, init_depth=2):
    return AutoformerTiny(num_classes=num_classes, embed_dim=init_width, depth=init_depth)
