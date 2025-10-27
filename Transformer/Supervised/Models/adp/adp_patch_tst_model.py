import torch, torch.nn as nn
from ADP_RetNet_model import copy_linear_overlap

# Treat each image row as a time series across width; patch along time axis
class RowTokenizer(nn.Module):
    def __init__(self, embed_dim=64, patch_len=4, img_h=32, in_ch=3):
        super().__init__()
        self.patch_len = patch_len
        self.embed_dim = embed_dim
        self.proj = nn.Conv1d(in_ch*img_h, embed_dim, kernel_size=patch_len, stride=patch_len)
    def forward(self, x):
        B,C,H,W = x.shape
        seq = x.permute(0,1,3,2).contiguous().view(B, C*H, W)  # B,(C*H),T
        tokens = self.proj(seq).transpose(1,2)                  # B,N,D
        return tokens

class TSTBlock(nn.Module):
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

class PatchTSTTiny(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=2, patch_len=4, nhead=4, img_h=32):
        super().__init__()
        self.embed_dim=embed_dim; self.patch_len=patch_len; self.nhead=nhead; self.img_h=img_h
        self.tokenizer = RowTokenizer(embed_dim, patch_len, img_h)
        self.blocks = nn.ModuleList([TSTBlock(embed_dim, nhead=nhead) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def add_block(self):
        self.blocks.append(TSTBlock(self.embed_dim, nhead=self.nhead))
    def widen_all(self, ex_k):
        new_dim = self.embed_dim + ex_k
        new_tok = RowTokenizer(new_dim, self.patch_len, self.img_h)
        # convolution copy overlap by weight slicing
        with torch.no_grad():
            oh, ow = self.tokenizer.proj.weight.shape
            nh, nw = new_tok.proj.weight.shape
            k = min(oh, nh); t = min(ow, nw)
            new_tok.proj.weight[:k,:t].copy_(self.tokenizer.proj.weight[:k,:t])
            if self.tokenizer.proj.bias is not None and new_tok.proj.bias is not None:
                new_tok.proj.bias[:min(len(new_tok.proj.bias), len(self.tokenizer.proj.bias))].copy_(self.tokenizer.proj.bias[:min(len(new_tok.proj.bias), len(self.tokenizer.proj.bias))])
        self.tokenizer = new_tok
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = TSTBlock(new_dim, nhead=min(self.nhead, max(1, new_dim//16)))
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


def build_patchtst(num_classes=10, init_width=64, init_depth=2, patch_len=4, img_h=32):
    return PatchTSTTiny(num_classes=num_classes, embed_dim=init_width, depth=init_depth, patch_len=patch_len, img_h=img_h)
