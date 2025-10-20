import torch
import torch.nn as nn
from typing import Tuple

# --- utility ---
def window_partition(x: torch.Tensor, window_size: int) -> torch.Tensor:
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    windows = x.permute(0,1,3,2,4,5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, H: int, W: int) -> torch.Tensor:
    B = int(windows.shape[0] // (H // window_size * W // window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0,1,3,2,4,5).contiguous().view(B, H, W, -1)
    return x

class MLP(nn.Module):
    def __init__(self, dim, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)
        self.drop = nn.Dropout(drop)
    def forward(self, x):
        x = self.fc1(x); x = self.act(x); x = self.drop(x); x = self.fc2(x); x = self.drop(x); return x

class WindowAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size: int, qkv_bias=True, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        # relative position bias
        ws = window_size
        coords = torch.stack(torch.meshgrid(torch.arange(ws), torch.arange(ws), indexing='ij'))  # 2, ws, ws
        coords_flat = coords.flatten(1)  # 2, ws*ws
        rel = coords_flat[:, :, None] - coords_flat[:, None, :]  # 2, N, N
        rel = rel.permute(1,2,0).contiguous()  # N, N, 2
        rel[:, :, 0] += ws - 1
        rel[:, :, 1] += ws - 1
        rel_index = rel[:, :, 0] * (2 * ws - 1) + rel[:, :, 1]
        self.register_buffer('relative_position_index', rel_index)
        self.relative_position_bias_table = nn.Parameter(torch.zeros(((2*ws-1)*(2*ws-1)), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

    def forward(self, x):
        # x: (B*nW, N, C) with N=window_size*window_size
        BnW, N, C = x.shape
        qkv = self.qkv(x).reshape(BnW, N, 3, self.num_heads, C // self.num_heads)
        q, k, v = qkv[:, :, 0], qkv[:, :, 1], qkv[:, :, 2]
        q = q * self.scale
        attn = q @ k.transpose(-2, -1)
        # add relative bias
        bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(N, N, -1)  # N,N,H
        attn = attn + bias.permute(2,0,1).unsqueeze(0)  # (1,H,N,N)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        out = (attn @ v).transpose(1,2).reshape(BnW, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        return out

class SwinBlock(nn.Module):
    def __init__(self, dim, num_heads, window_size=4, shift=False, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.window_size = window_size
        self.shift = shift
        self.norm1 = nn.LayerNorm(dim)
        self.attn = WindowAttention(dim, num_heads, window_size)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)

    def forward(self, x: torch.Tensor, H: int, W: int) -> Tuple[torch.Tensor, int, int]:
        B, L, C = x.shape
        assert L == H * W
        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)
        if self.shift:
            shift = self.window_size // 2
            x = torch.roll(x, shifts=(-shift, -shift), dims=(1,2))
        # partition windows
        x_windows = window_partition(x, self.window_size)  # (BnW, ws, ws, C)
        x_windows = x_windows.view(-1, self.window_size*self.window_size, C)
        # attention
        attn_windows = self.attn(x_windows)
        # merge
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        x = window_reverse(attn_windows, self.window_size, H, W)
        if self.shift:
            shift = self.window_size // 2
            x = torch.roll(x, shifts=(shift, shift), dims=(1,2))
        x = x.view(B, H*W, C)
        x = shortcut + x
        # MLP
        x = x + self.mlp(self.norm2(x))
        return x, H, W

class PatchEmbed(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=4):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
    def forward(self, x):
        x = self.proj(x)  # B, C, H', W'
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)  # B, HW, C
        return x, H, W

class PatchMerging(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.reduction = nn.Linear(4*dim, 2*dim)
        self.norm = nn.LayerNorm(4*dim)
    def forward(self, x, H, W):
        B, L, C = x.shape
        x = x.view(B, H, W, C)
        # merge 2x2
        x0 = x[:, 0::2, 0::2, :]
        x1 = x[:, 1::2, 0::2, :]
        x2 = x[:, 0::2, 1::2, :]
        x3 = x[:, 1::2, 1::2, :]
        x = torch.cat([x0, x1, x2, x3], dim=-1).view(B, -1, 4*C)
        x = self.norm(x)
        x = self.reduction(x)
        H, W = H // 2, W // 2
        return x, H, W

class SwinTiny(nn.Module):
    """A compact Swin-like backbone for CIFAR-10/100 classification.
    Stages: [2,2,6,2] blocks, dims [96,192,384,768] by default (scaled down here).
    """
    def __init__(self, num_classes=10, in_chans=3, img_size=32,
                 embed_dim=96, depths=(2,2,6,2), num_heads=(3,6,12,24), window_size=4, mlp_ratio=4.0):
        super().__init__()
        self.patch_embed = PatchEmbed(in_chans, embed_dim, patch_size=4)
        self.pos_drop = nn.Dropout(0.0)

        self.stages = nn.ModuleList()
        dims = [embed_dim, embed_dim*2, embed_dim*4, embed_dim*8]
        for stage_idx, depth in enumerate(depths):
            blocks = []
            for i in range(depth):
                shift = (i % 2 == 1)
                blocks.append(SwinBlock(dims[stage_idx], num_heads[stage_idx], window_size, shift, mlp_ratio))
            self.stages.append(nn.Sequential(*blocks))
            if stage_idx < len(depths) - 1:
                self.stages.append(PatchMerging(dims[stage_idx]))

        self.norm = nn.LayerNorm(dims[-1])
        self.head = nn.Linear(dims[-1], num_classes)

    def forward(self, x):
        x, H, W = self.patch_embed(x)  # B, HW, C
        x = self.pos_drop(x)
        stage_id = 0
        while stage_id < len(self.stages):
            blk = self.stages[stage_id]
            if isinstance(blk, nn.Sequential):
                for b in blk:
                    x, H, W = b(x, H, W)
            else:  # PatchMerging
                x, H, W = blk(x, H, W)
            stage_id += 1
        x = self.norm(x)
        x = x.mean(dim=1)
        return self.head(x)
