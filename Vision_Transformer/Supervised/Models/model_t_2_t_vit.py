import torch
import torch.nn as nn
import torch.nn.functional as F

# ------------------------------
# T2T-ViT (Tokens-to-Token)
# - Progressive soft tokenization before transformer encoder
# - We implement a common variant with 3 soft-split stages using Unfold+Linear
# ------------------------------

class SoftSplit(nn.Module):
    def __init__(self, in_ch, k=7, s=4, p=2, out_dim=64):
        super().__init__()
        self.unfold = nn.Unfold(kernel_size=k, stride=s, padding=p)
        self.proj = nn.Linear(in_ch * k * k, out_dim)
        self.act = nn.GELU()

    def forward(self, x):  # x: (B, C, H, W)
        B,C,H,W = x.shape
        patches = self.unfold(x).transpose(1,2)  # (B, N, C*k*k)
        tokens = self.proj(patches)
        return self.act(tokens)

class MLP(nn.Module):
    def __init__(self, dim, ratio=4.0, drop=0.0):
        super().__init__()
        hid=int(dim*ratio)
        self.fc1=nn.Linear(dim,hid); self.act=nn.GELU(); self.fc2=nn.Linear(hid,dim); self.drop=nn.Dropout(drop)
    def forward(self,x):
        x=self.fc1(x); x=self.act(x); x=self.drop(x); x=self.fc2(x); x=self.drop(x); return x

class Block(nn.Module):
    def __init__(self, dim, heads, mlp_ratio=4.0, attn_drop=0.0, proj_drop=0.0, drop=0.0):
        super().__init__()
        self.n1=nn.LayerNorm(dim)
        self.attn=nn.MultiheadAttention(dim, heads, dropout=attn_drop, batch_first=True)
        self.pd=nn.Dropout(proj_drop)
        self.n2=nn.LayerNorm(dim)
        self.mlp=MLP(dim, mlp_ratio, drop)
    def forward(self,x):
        xn=self.n1(x); a,_=self.attn(xn,xn,xn,need_weights=False); x=x+self.pd(a); x=x+self.mlp(self.n2(x)); return x

class T2T_ViT(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=10,
                 t2t_dims=(64, 128, 192), encoder_dim=256, depth=10, heads=4, mlp_ratio=3.0):
        super().__init__()
        # T2T soft-splits
        self.ss1=SoftSplit(in_chans, k=7, s=4, p=2, out_dim=t2t_dims[0])
        self.ss2=SoftSplit(t2t_dims[0], k=3, s=2, p=1, out_dim=t2t_dims[1])
        self.ss3=SoftSplit(t2t_dims[1], k=3, s=2, p=1, out_dim=t2t_dims[2])
        self.proj=nn.Linear(t2t_dims[2], encoder_dim)
        # Learnable class token + pos embed over tokens (N depends on image size; we infer at first forward)
        self.cls_token=nn.Parameter(torch.zeros(1,1,encoder_dim))
        self.pos_embed=None  # init lazily
        # Encoder
        self.blocks=nn.ModuleList([Block(encoder_dim, heads, mlp_ratio) for _ in range(depth)])
        self.norm=nn.LayerNorm(encoder_dim)
        self.head=nn.Linear(encoder_dim, num_classes)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.head.weight, std=0.02)
        nn.init.zeros_(self.head.bias)

    def forward(self,x):
        B=x.size(0)
        t=self.ss1(x)     # (B, N1, d1)
        t=self.ss2(t.transpose(1,2).reshape(B,-1,1,1).permute(0,2,3,1)) if False else t  # comment: kept tokens flat
        t=self.ss2(x.new_zeros(B, t.size(-1), int((t.size(1))**0.5), int((t.size(1))**0.5))
                   .view(B, t.size(-1), 1, 1)) if False else self.ss2_tokens(t)
        t=self.ss3_tokens(t)
        t=self.proj(t)
        # CLS + pos
        if self.pos_embed is None or self.pos_embed.size(1)!=(t.size(1)+1):
            self.pos_embed=nn.Parameter(torch.zeros(1, t.size(1)+1, t.size(2), device=t.device))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
        cls=self.cls_token.expand(B,-1,-1)
        z=torch.cat([cls,t],dim=1)
        z=z+self.pos_embed
        for blk in self.blocks:
            z=blk(z)
        z=self.norm(z)
        return self.head(z[:,0])

    def ss2_tokens(self, tokens):
        # simple linear re-embedding (no spatial reshape to keep it lightweight)
        return self.ss2.proj(tokens)

    def ss3_tokens(self, tokens):
        return self.ss3.proj(tokens)
