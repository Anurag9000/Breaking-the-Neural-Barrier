import torch, torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        x = self.norm(self.proj(x))
        B,C,H,W = x.shape
        return x.flatten(2).transpose(1,2)

class MixerBlock(nn.Module):
    def __init__(self, num_tokens, dim, token_mlp_ratio=0.5, channel_mlp_ratio=4, drop=0.0):
        super().__init__()
        t_hidden = max(1, int(num_tokens*token_mlp_ratio))
        c_hidden = int(dim*channel_mlp_ratio)
        self.norm1 = nn.LayerNorm(dim)
        self.token_mlp = nn.Sequential(
            nn.Linear(num_tokens, t_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(t_hidden, num_tokens), nn.Dropout(drop)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, c_hidden), nn.GELU(), nn.Dropout(drop),
            nn.Linear(c_hidden, dim), nn.Dropout(drop)
        )
    def forward(self, x):
        # token mixing: act on transpose
        y = self.norm1(x)
        y = y.transpose(1,2)
        y = self.token_mlp(y)
        y = y.transpose(1,2)
        x = x + y
        x = x + self.channel_mlp(self.norm2(x))
        return x

class MLPMixerTiny(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=4, patch=4, image_size=32):
        super().__init__()
        self.embed_dim = embed_dim; self.patch=patch
        self.grid = (image_size//patch)*(image_size//patch)
        self.tokenizer = PatchEmbed(3, embed_dim, patch)
        self.blocks = nn.ModuleList([MixerBlock(self.grid, embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def add_block(self):
        self.blocks.append(MixerBlock(self.grid, self.embed_dim))
    def widen_all(self, ex_k):
        new_dim = self.embed_dim + ex_k
        new_tok = PatchEmbed(3, new_dim, self.patch)
        copy_conv2d(self.tokenizer.proj, new_tok.proj)
        copy_bn2d(self.tokenizer.norm, new_tok.norm)
        self.tokenizer = new_tok
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = MixerBlock(self.grid, new_dim)
            transplant_block_mixer(b, nb)
            new_blocks.append(nb)
        self.blocks = new_blocks
        self.norm = nn.LayerNorm(new_dim)
        new_head = nn.Linear(new_dim, self.head.out_features)
        copy_linear_overlap(self.head, new_head)
        self.head = new_head
        self.embed_dim = new_dim
    def forward(self, x):
        x = self.tokenizer(x)
        for b in self.blocks: x = b(x)
        x = self.norm(x).mean(1)
        return self.head(x)

from ADP_RetNet_model import copy_conv2d, copy_bn2d, copy_linear_overlap, TrainCfg, ADPCfg, evaluate, train_inner
import torch.nn as nn
@torch.no_grad()
def transplant_block_mixer(old: MixerBlock, new: MixerBlock):
    # token mlp
    for (ol, nl) in zip(old.token_mlp, new.token_mlp):
        if isinstance(ol, nn.Linear) and isinstance(nl, nn.Linear):
            copy_linear_overlap(ol, nl)
    for (ol, nl) in zip(old.channel_mlp, new.channel_mlp):
        if isinstance(ol, nn.Linear) and isinstance(nl, nn.Linear):
            copy_linear_overlap(ol, nl)

def build_mlp_mixer(num_classes=10, init_width=64, init_depth=2, patch=4, image_size=32):
    return MLPMixerTiny(num_classes=num_classes, embed_dim=init_width, depth=init_depth, patch=patch, image_size=image_size)
