import torch
import torch.nn as nn
from model_mae_vit import PatchEmbed, TransformerEncoder

class CAEViT(nn.Module):
    """Masked image modeling with latent regression (CAE-style)."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=384, depth=6, heads=6, mlp_ratio=4.0, mask_ratio=0.6,
                 latent_dim=256):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.encoder = TransformerEncoder(embed_dim, depth, heads, mlp_ratio)
        self.latent_proj = nn.Linear(embed_dim, latent_dim)
        self.head = nn.Linear(embed_dim, latent_dim)

    def random_mask(self, x):
        B, N, D = x.shape
        num_mask = int(self.mask_ratio * N)
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        ids_keep = ids_shuffle[:, num_mask:]
        mask = torch.ones(B, N, device=x.device); mask[:, num_mask:] = 0
        mask = torch.gather(mask, 1, ids_restore)
        return ids_keep, mask

    def forward(self, imgs):
        x = self.patch(imgs) + self.pos
        ids_keep, mask = self.random_mask(x)
        B, N, D = x.shape
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1,1,D))
        h = self.encoder(x_vis)
        # target latents (stop-grad)
        with torch.no_grad():
            t_lat = self.latent_proj(x)  # pre-encoded latent targets
            t_lat = (t_lat - t_lat.mean(dim=-1, keepdim=True)) / (t_lat.std(dim=-1, keepdim=True) + 1e-6)
        # scatter preds back
        out = torch.zeros(B, N, D, device=imgs.device)
        out.scatter_(1, ids_keep.unsqueeze(-1).repeat(1,1,D), h)
        pred_lat = self.head(out)
        pred_lat = (pred_lat - pred_lat.mean(dim=-1, keepdim=True)) / (pred_lat.std(dim=-1, keepdim=True) + 1e-6)
        return pred_lat, t_lat, mask

    def loss(self, outputs):
        pred_lat, t_lat, mask = outputs
        loss = (pred_lat - t_lat).pow(2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss
