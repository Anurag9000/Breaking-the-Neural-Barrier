import torch
import torch.nn as nn
import torch.nn.functional as F

class Patchify(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=256):
        super().__init__()
        self.ps = patch_size
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x).flatten(2).transpose(1,2)  # (B,N,D)
        return x

class CausalBlock(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Linear(int(dim*mlp_ratio), dim))
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        B,N,D = x.shape
        mask = torch.triu(torch.ones(N,N, device=x.device), diagonal=1).bool()
        x = x + self.drop(self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)[0])
        x = x + self.drop(self.mlp(self.ln2(x)))
        return x

class iGPTPatch(nn.Module):
    def __init__(self, img_size=32, patch_size=4, in_ch=3, embed_dim=256, depth=8, heads=8, vocab_size=8192):
        super().__init__()
        self.patch = Patchify(img_size, patch_size, in_ch, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.blocks = nn.ModuleList([CausalBlock(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size)
        # simple vector-quantizer codebook for patch tokens (frozen)
        self.register_buffer('codebook', torch.randn(vocab_size, embed_dim))

    @torch.no_grad()
    def encode_tokens(self, x):
        # x: (B,N,D) -> nearest code index per patch
        d = torch.cdist(x.reshape(-1, x.size(-1)), self.codebook)
        idx = d.argmin(dim=1).view(x.size(0), x.size(1))
        return idx

    def forward(self, imgs):
        x = self.patch(imgs) + self.pos
        # targets = next-token indices from codebook
        with torch.no_grad():
            idx = self.encode_tokens(self.patch(imgs))
        B,N,D = x.shape
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        logits = self.head(x)  # (B,N,V)
        # language-model loss over sequence of patches
        logits = logits[:, :-1, :].contiguous()
        targets = idx[:, 1:].contiguous()
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return loss
