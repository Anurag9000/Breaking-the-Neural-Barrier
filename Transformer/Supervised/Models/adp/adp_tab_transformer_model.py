import torch, torch.nn as nn
from ADP_RetNet_model import copy_conv2d, copy_bn2d, copy_linear_overlap

# We emulate TabTransformer's feature-column attention by treating each image patch as a 'column'
# and using attention across columns; practical for CIFAR-10 patch tokens.

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        x = self.norm(self.proj(x))
        B,D,H,W = x.shape
        return x.flatten(2).transpose(1,2)  # B,N,D (N ~ columns)

class ColumnAttnBlock(nn.Module):
    def __init__(self, dim, nhead=4, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(nn.Linear(dim, mlp_ratio*dim), nn.GELU(), nn.Dropout(drop), nn.Linear(mlp_ratio*dim, dim), nn.Dropout(drop))
    def forward(self, x):
        h = self.norm1(x)
        y,_ = self.attn(h,h,h, need_weights=False)
        x = x + self.drop(y)
        x = x + self.ff(self.norm2(x))
        return x

class TabTransformerTiny(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=2, patch=4, nhead=4):
        super().__init__()
        self.embed_dim=embed_dim; self.patch=patch; self.nhead=nhead
        self.tokenizer = PatchEmbed(3, embed_dim, patch)
        self.blocks = nn.ModuleList([ColumnAttnBlock(embed_dim, nhead=nhead) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def add_block(self):
        self.blocks.append(ColumnAttnBlock(self.embed_dim, nhead=self.nhead))
    def widen_all(self, ex_k):
        new_dim = self.embed_dim + ex_k
        new_tok = PatchEmbed(3, new_dim, self.patch)
        copy_conv2d(self.tokenizer.proj, new_tok.proj)
        copy_bn2d(self.tokenizer.norm, new_tok.norm)
        self.tokenizer = new_tok
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = ColumnAttnBlock(new_dim, nhead=min(self.nhead, max(1, new_dim//16)))
            # copy FF overlap
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


def build_tabtransformer(num_classes=10, init_width=64, init_depth=2, patch=4):
    return TabTransformerTiny(num_classes=num_classes, embed_dim=init_width, depth=init_depth, patch=patch)
