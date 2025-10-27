import torch, torch.nn as nn
from dataclasses import dataclass

# MetaFormer / EfficientFormer-style: Token Mixer (DWConv) + MLP, no attention
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed_dim=64, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed_dim, kernel_size=patch, stride=patch)
        self.norm = nn.BatchNorm2d(embed_dim)
    def forward(self, x):
        x = self.norm(self.proj(x))
        B,C,H,W = x.shape
        return x.flatten(2).transpose(1,2)

class TokenMixerDW(nn.Module):
    def __init__(self, dim, k=3):
        super().__init__()
        self.dw = nn.Conv1d(dim, dim, kernel_size=k, padding=k//2, groups=dim)
    def forward(self, x):
        y = self.dw(x.transpose(1,2)).transpose(1,2)
        return y

class MFBlock(nn.Module):
    def __init__(self, dim, mlp_ratio=4, drop=0.0, k=3):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.mixer = TokenMixerDW(dim, k)
        self.drop = nn.Dropout(drop)
        self.ff = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, mlp_ratio*dim), nn.GELU(), nn.Dropout(drop),
            nn.Linear(mlp_ratio*dim, dim), nn.Dropout(drop)
        )
    def forward(self, x):
        x = x + self.drop(self.mixer(self.norm1(x)))
        x = x + self.ff(x)
        return x

class MetaFormerTiny(nn.Module):
    def __init__(self, num_classes=10, embed_dim=64, depth=4, patch=4):
        super().__init__()
        self.embed_dim = embed_dim; self.patch=patch
        self.tokenizer = PatchEmbed(3, embed_dim, patch)
        self.blocks = nn.ModuleList([MFBlock(embed_dim) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
    def add_block(self):
        self.blocks.append(MFBlock(self.embed_dim))
    def widen_all(self, ex_k):
        new_dim = self.embed_dim + ex_k
        new_tok = PatchEmbed(3, new_dim, self.patch)
        copy_conv2d(self.tokenizer.proj, new_tok.proj)
        copy_bn2d(self.tokenizer.norm, new_tok.norm)
        self.tokenizer = new_tok
        new_blocks = nn.ModuleList()
        for b in self.blocks:
            nb = MFBlock(new_dim)
            transplant_block_mf(b, nb)
            new_blocks.append(nb)
        self.blocks = new_blocks
        self.norm = nn.LayerNorm(new_dim)
        new_head = nn.Linear(new_dim, self.head.out_features)
        copy_linear_overlap(self.head, new_head)
        self.head = new_head
        self.embed_dim = new_dim
    def forward(self, x):
        x = self.tokenizer(x)
        for b in self.blocks:
            x = b(x)
        x = self.norm(x).mean(1)
        return self.head(x)

# overlap-copy helpers
import torch.nn.functional as F
@torch.no_grad()
def copy_conv2d(old, new):
    new.weight.zero_()
    oh, ow = old.weight.shape[:2]
    kh, kw = min(new.weight.shape[2], old.weight.shape[2]), min(new.weight.shape[3], old.weight.shape[3])
    new.weight[:oh,:ow,:kh,:kw].copy_(old.weight[:oh,:ow,:kh,:kw])
    if old.bias is not None and new.bias is not None:
        new.bias[:old.bias.numel()].copy_(old.bias)
@torch.no_grad()
def copy_bn2d(old, new):
    c = min(old.num_features, new.num_features)
    new.weight[:c].copy_(old.weight[:c]); new.bias[:c].copy_(old.bias[:c])
    new.running_mean[:c].copy_(old.running_mean[:c]); new.running_var[:c].copy_(old.running_var[:c])
@torch.no_grad()
def copy_linear_overlap(old, new):
    out = min(old.out_features, new.out_features); inn=min(old.in_features, new.in_features)
    new.weight[:out,:inn].copy_(old.weight[:out,:inn])
    if old.bias is not None and new.bias is not None:
        new.bias[:out].copy_(old.bias[:out])
@torch.no_grad()
def transplant_block_mf(old: MFBlock, new: MFBlock):
    # mixer conv weights
    oc = min(old.mixer.dw.weight.shape[0], new.mixer.dw.weight.shape[0])
    k = min(old.mixer.dw.weight.shape[-1], new.mixer.dw.weight.shape[-1])
    new.mixer.dw.weight[:oc,0,:k].copy_(old.mixer.dw.weight[:oc,0,:k])
    for (ol, nl) in zip(old.ff, new.ff):
        if isinstance(ol, nn.Linear) and isinstance(nl, nn.Linear):
            copy_linear_overlap(ol, nl)

# minimal train/eval + ADP orchestrator shared with RetNet
from ADP_RetNet_model import TrainCfg, ADPCfg, EarlyStopper, evaluate, train_inner

def build_metaformer(num_classes=10, init_width=64, init_depth=2, patch=4):
    return MetaFormerTiny(num_classes=num_classes, embed_dim=init_width, depth=init_depth, patch=patch)
