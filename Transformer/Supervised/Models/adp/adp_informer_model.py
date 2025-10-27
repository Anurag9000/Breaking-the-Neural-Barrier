import torch, torch.nn as nn
from ADP_RetNet_model import copy_linear_overlap

# Simplified ProbSparse: attend to a subsampled set of keys (stride) to simulate Informer efficiency
class SeqTokenizer(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, seq_len=32*32):
        super().__init__()
        self.proj = nn.Linear(in_ch, embed_dim)
        self.seq_len = seq_len
    def forward(self, x):
        B,C,H,W = x.shape
        seq = x.view(B, C, H*W).transpose(1,2)  # B,T,C
        return self.proj(seq)

class SubsampledMHA(nn.Module):
    def __init__(self, dim, nhead=4, stride=4, drop=0.0):
        super().__init__()
        self.mha = nn.MultiheadAttention(dim, nhead, batch_first=True, dropout=drop)
        self.stride = stride
    def forward(self, x):
        # Subsample keys/values by stride
        k = x[:, ::self.stride, :]
        v = x[:, ::self.stride, :]
        y,_ = self.mha(x, k, v, need_weights=False)
        return y

class InformerBlock(nn.Module):
    def __init__(self, dim, nhead=4, stride=4, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SubsampledMHA(dim, nhead=nhead, stride=stride, drop=drop)
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, mlp_ratio*dim), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp_ratio*dim, dim), nn.Dropout(drop))
    def forward(self, x):
        x = x + self.drop(self.attn(self.norm1(x)))
        x = x + self.ff(self.norm2(x))
        return x

class InformerTiny(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=2, nhead=4, stride=4):
        super().__init__()
        self.embed_dim = embed_dim; self.nhead=nhead; self.stride=stride
        self.tokenizer = SeqTokenizer(in_ch=3, embed_dim=embed_dim)
        self.blocks = nn.ModuleList([InformerBlock(embed_dim, nhead=nhead, stride=stride) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def add_block(self):
        self.blocks.append(InformerBlock(self.embed_dim, nhead=self.nhead, stride=self.stride))
    def widen_all(self, ex_k):
        new_dim = self.embed_dim + ex_k
        new_tok = SeqTokenizer(in_ch=3, embed_dim=new_dim)
        with torch.no_grad():
            copy_linear_overlap(self.tokenizer.proj, new_tok.proj)
        self.tokenizer = new_tok
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = InformerBlock(new_dim, nhead=min(self.nhead, max(1, new_dim//16)), stride=self.stride)
            for (ol, nl) in zip(b.ff, nb.ff):
                if isinstance(ol, nn.Linear) and isinstance(nl, nn.Linear): copy_linear_overlap(ol, nl)
            new_blocks.append(nb)
        self.blocks = new_blocks
        self.norm = nn.LayerNorm(new_dim)
        new_head = nn.Linear(new_dim, self.head.out_features); copy_linear_overlap(self.head, new_head); self.head = new_head
        self.embed_dim = new_dim
    def forward(self, x):
        x = self.tokenizer(x)
        for b in self.blocks: x = b(x)
        x = self.norm(x).mean(1)
        return self.head(x)


def build_informer(num_classes=10, init_width=64, init_depth=2):
    return InformerTiny(num_classes=num_classes, embed_dim=init_width, depth=init_depth)
