import torch, torch.nn as nn
from ADP_RetNet_model import copy_linear_overlap

# FEDformer: frequency enhanced; keep top-K low-frequency modes via FFT and MLP refine
class FEDBlock(nn.Module):
    def __init__(self, dim, topk=16, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.topk = topk
        self.norm = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, mlp_ratio*dim), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp_ratio*dim, dim), nn.Dropout(drop))
    def forward(self, x):
        h = self.norm(x)
        H = torch.fft.rfft(h, dim=1)
        # keep top-k low frequency indices
        k = min(self.topk, H.size(1))
        mask = torch.zeros_like(H)
        mask[:, :k, :] = 1
        Hf = H * mask
        y = torch.fft.irfft(Hf, n=h.size(1), dim=1).real
        x = x + y
        x = x + self.ff(x)
        return x

class FEDformerTiny(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=2, topk=16):
        super().__init__()
        self.embed_dim=embed_dim; self.topk=topk
        self.tokenizer = nn.Linear(3, embed_dim)
        self.blocks = nn.ModuleList([FEDBlock(embed_dim, topk=topk) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def add_block(self):
        self.blocks.append(FEDBlock(self.embed_dim, topk=self.topk))
    def widen_all(self, ex_k):
        new_dim = self.embed_dim + ex_k
        new_tok = nn.Linear(3, new_dim)
        with torch.no_grad(): copy_linear_overlap(self.tokenizer, new_tok)
        self.tokenizer = new_tok
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = FEDBlock(new_dim, topk=self.topk)
            for (ol, nl) in zip(b.ff, nb.ff):
                if isinstance(ol, nn.Linear) and isinstance(nl, nn.Linear): copy_linear_overlap(ol, nl)
            new_blocks.append(nb)
        self.blocks = new_blocks
        self.norm = nn.LayerNorm(new_dim)
        new_head = nn.Linear(new_dim, self.head.out_features); copy_linear_overlap(self.head, new_head); self.head = new_head
        self.embed_dim = new_dim
    def forward(self, x):
        B,C,H,W = x.shape
        x = x.view(B,C,H*W).transpose(1,2)   # B,N,3
        x = self.tokenizer(x)
        for b in self.blocks: x = b(x)
        x = self.norm(x).mean(1)
        return self.head(x)


def build_fedformer(num_classes=10, init_width=64, init_depth=2, topk=16):
    return FEDformerTiny(num_classes=num_classes, embed_dim=init_width, depth=init_depth, topk=topk)
