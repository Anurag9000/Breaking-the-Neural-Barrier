import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

# =====================
# ViT backbone (minimal)
# =====================
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=384):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid = img_size // patch_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)  # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, dim=384, depth=6, heads=6, mlp_ratio=4.0, dropout=0.0):
        super().__init__()
        layers = []
        for _ in range(depth):
            layers.append(nn.ModuleList([
                nn.LayerNorm(dim),
                nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True),
                nn.LayerNorm(dim),
                nn.Sequential(
                    nn.Linear(dim, int(dim * mlp_ratio)),
                    nn.GELU(),
                    nn.Linear(int(dim * mlp_ratio), dim)
                )
            ]))
        self.layers = nn.ModuleList(layers)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):  # (B, N, D)
        for ln1, attn, ln2, mlp in self.layers:
            x = x + self.dropout(attn(ln1(x), ln1(x), ln1(x), need_weights=False)[0])
            x = x + self.dropout(mlp(ln2(x)))
        return x

# =====================
# MAE modules
# =====================
class MAEViT(nn.Module):
    """Masked Autoencoder with ViT encoder and lightweight decoder.
    - Encoder sees only visible patches.
    - Decoder reconstructs pixels for all patches at masked locations.
    """
    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=384,
                 depth=6,
                 heads=6,
                 mlp_ratio=4.0,
                 dec_embed_dim=192,
                 dec_depth=4,
                 dec_heads=6,
                 mask_ratio=0.75):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos_enc = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, embed_dim))
        nn.init.trunc_normal_(self.pos_enc, std=0.02)
        self.encoder = TransformerEncoder(embed_dim, depth, heads, mlp_ratio)
        # projector to decoder dim
        self.enc_to_dec = nn.Linear(embed_dim, dec_embed_dim)
        # decoder tokens and transformer
        self.mask_token = nn.Parameter(torch.zeros(1, 1, dec_embed_dim))
        nn.init.trunc_normal_(self.mask_token, std=0.02)
        self.dec_pos = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, dec_embed_dim))
        nn.init.trunc_normal_(self.dec_pos, std=0.02)
        self.decoder = TransformerEncoder(dec_embed_dim, dec_depth, dec_heads, 4.0)
        # pixel reconstruction head
        self.patch_size = patch_size
        patch_pixels = patch_size * patch_size * in_chans
        self.head = nn.Linear(dec_embed_dim, patch_pixels)

    def random_mask(self, x):
        B, N, D = x.shape
        num_mask = int(self.mask_ratio * N)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, num_mask:]
        x_keep = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1,1,D))
        mask = torch.ones(B, N, device=x.device)
        mask[:, num_mask:] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return x_keep, mask, ids_restore, ids_keep

    def forward(self, imgs):
        # 1) patch + add pos
        x = self.patch(imgs)
        x = x + self.pos_enc
        B, N, D = x.shape
        # 2) random mask
        x_vis, mask, ids_restore, ids_keep = self.random_mask(x)
        # 3) encode visible tokens
        x_enc = self.encoder(x_vis)
        # 4) map to decoder
        x_dec_vis = self.enc_to_dec(x_enc)
        # 5) prepare full sequence with mask tokens
        B, Nv, De = x_dec_vis.shape
        mask_tokens = self.mask_token.repeat(B, N - Nv, 1)
        # scatter back
        x_dec_full = torch.zeros(B, N, De, device=imgs.device)
        x_dec_full.scatter_(1, ids_keep.unsqueeze(-1).repeat(1,1,De), x_dec_vis)
        x_dec_full[mask.bool()] = mask_tokens.reshape(-1, De)
        # 6) add decoder pos + decode
        x_dec = x_dec_full + self.dec_pos
        x_dec = self.decoder(x_dec)
        # 7) predict pixels
        pred = self.head(x_dec)
        return pred, mask

    def loss(self, pred, imgs):
        # target pixels
        B, C, H, W = imgs.shape
        ps = self.patch_size
        target = imgs.unfold(2, ps, ps).unfold(3, ps, ps)  # (B,C,Hp,Wp,ps,ps)
        target = target.permute(0,2,3,1,4,5).contiguous()  # (B,Hp,Wp,C,ps,ps)
        target = target.view(B, -1, C*ps*ps)  # (B, N, PP)
        pred, mask = pred
        # L2 only on masked tokens
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss
