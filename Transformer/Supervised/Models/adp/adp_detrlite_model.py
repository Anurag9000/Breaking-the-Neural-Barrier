import torch, torch.nn as nn
from dataclasses import dataclass

# Very small DETR-like encoder + set head, adapted for classification by using 1 query
# (keeps ADP growth semantics on width (embed dim) and depth (num encoder blocks))

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        x = self.norm(self.proj(x))
        B,C,H,W = x.shape
        return x.flatten(2).transpose(1,2) # B,N,D

class EncoderBlock(nn.Module):
    def __init__(self, dim, nhead=4, mlp_ratio=4, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, nhead, dropout=drop, batch_first=True)
        self.drop = nn.Dropout(drop)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = nn.Sequential(
            nn.Linear(dim, mlp_ratio*dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_ratio*dim, dim), nn.Dropout(drop)
        )
    def forward(self, x):
        h = self.norm1(x)
        y,_ = self.attn(h,h,h, need_weights=False)
        x = x + self.drop(y)
        x = x + self.ff(self.norm2(x))
        return x

class DETRLite(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=2, patch=4, nhead=4):
        super().__init__()
        self.embed_dim = embed_dim; self.patch=patch; self.nhead = nhead
        self.tokenizer = PatchEmbed(3, embed_dim, patch)
        self.blocks = nn.ModuleList([EncoderBlock(embed_dim, nhead=nhead) for _ in range(depth)])
        self.query = nn.Parameter(torch.randn(1,1,embed_dim))  # 1 query for classification
        self.head = nn.Linear(embed_dim, num_classes)
        self.norm = nn.LayerNorm(embed_dim)
    def add_block(self):
        self.blocks.append(EncoderBlock(self.embed_dim, nhead=self.nhead))
    def widen_all(self, ex_k):
        new_dim = self.embed_dim + ex_k
        new_tok = PatchEmbed(3, new_dim, self.patch)
        copy_conv2d(self.tokenizer.proj, new_tok.proj)
        copy_bn2d(self.tokenizer.norm, new_tok.norm)
        self.tokenizer = new_tok
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = EncoderBlock(new_dim, nhead=min(self.nhead, max(1, new_dim//16)))
            transplant_block_enc(b, nb)
            new_blocks.append(nb)
        self.blocks = new_blocks
        self.norm = nn.LayerNorm(new_dim)
        new_head = nn.Linear(new_dim, self.head.out_features)
        copy_linear_overlap(self.head, new_head)
        self.head = new_head
        with torch.no_grad():
            q = torch.zeros(1,1,new_dim); q[:,:,:self.embed_dim].copy_(self.query)
            self.query = nn.Parameter(q)
        self.embed_dim = new_dim
    def forward(self, x):
        x = self.tokenizer(x)
        for b in self.blocks:
            x = b(x)
        B = x.size(0)
        q = self.query.expand(B, -1, -1)
        # attend query over encoded tokens
        q = EncoderBlock(self.embed_dim, nhead=self.nhead)(torch.cat([q,x], dim=1))[:,0:1,:]
        q = self.norm(q).squeeze(1)
        return self.head(q)

from ADP_RetNet_model import copy_conv2d, copy_bn2d, copy_linear_overlap, TrainCfg, ADPCfg, evaluate, train_inner

@torch.no_grad()
def transplant_block_enc(old: EncoderBlock, new: EncoderBlock):
    # Copy FFN overlap; MHA can't be trivially overlapped across dim changes, so skip
    for (ol, nl) in zip(old.ff, new.ff):
        if isinstance(ol, nn.Linear) and isinstance(nl, nn.Linear):
            copy_linear_overlap(ol, nl)

def build_detrlite(num_classes=10, init_width=64, init_depth=2, patch=4):
    return DETRLite(num_classes=num_classes, embed_dim=init_width, depth=init_depth, patch=patch)
