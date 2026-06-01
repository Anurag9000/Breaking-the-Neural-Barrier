import torch
import torch.nn as nn
from model_mae_vit import PatchEmbed, TransformerEncoder

class SimMIMViT(nn.Module):
    """SimMIM: regress raw pixels of masked patches (single-model)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=384, depth=6, heads=6, mlp_ratio=4.0, mask_ratio=0.6):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.encoder = TransformerEncoder(embed_dim, depth, heads, mlp_ratio)
        self.patch_size = patch_size
        self.head = nn.Linear(embed_dim, in_chans * patch_size * patch_size)

    def random_mask(self, x):
        B, N, D = x.shape
        num_mask = int(self.mask_ratio * N)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, num_mask:]
        mask = torch.ones(B, N, device=x.device);
        mask[:, num_mask:] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return ids_keep, mask

    def forward(self, imgs):
        x = self.patch(imgs) + self.pos
        ids_keep, mask = self.random_mask(x)
        B, N, D = x.shape
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1,1,D))
        h = self.encoder(x_vis)
        # scatter back to N positions via zeros (only compute loss on masked tokens)
        out = torch.zeros(B, N, D, device=imgs.device)
        out.scatter_(1, ids_keep.unsqueeze(-1).repeat(1,1,D), h)
        pred = self.head(out)
        return pred, mask

    def loss(self, pred_tuple, imgs):
        pred, mask = pred_tuple
        B, C, H, W = imgs.shape; ps = self.patch_size
        target = imgs.unfold(2, ps, ps).unfold(3, ps, ps)
        target = target.permute(0,2,3,1,4,5).contiguous().view(B, -1, C*ps*ps)
        loss = ((pred - target) ** 2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss
