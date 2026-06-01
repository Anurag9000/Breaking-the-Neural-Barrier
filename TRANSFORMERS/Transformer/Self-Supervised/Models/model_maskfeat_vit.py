import torch
import torch.nn as nn
import torch.nn.functional as F
from model_mae_vit import PatchEmbed, TransformerEncoder

def simple_hog(patches, bins=8, eps=1e-6):
    """Compute a tiny HOG-like descriptor per patch (B,N,Cpsps)->(B,N,bins).
    We use Sobel filters and pool orientations into bins.
    """
    B, N, PP = patches.shape
    C = 3
    ps = int((PP // C) ** 0.5)
    x = patches.view(B*N, C, ps, ps)
    # Sobel
    kx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    ky = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    gx = F.conv2d(x, kx.repeat(C,1,1,1), groups=C, padding=1)
    gy = F.conv2d(x, ky.repeat(C,1,1,1), groups=C, padding=1)
    mag = torch.sqrt(gx**2 + gy**2 + eps)
    ang = torch.atan2(gy, gx)  # [-pi, pi]
    ang = (ang + torch.pi) / (2*torch.pi)  # [0,1)
    # binning
    h = torch.zeros(x.size(0), bins, device=x.device)
    bin_idx = (ang * bins).long().clamp(max=bins-1)
    for b in range(bins):
        h[:, b] = mag[bin_idx==b].view(x.size(0), -1).sum(dim=1)
    h = h / (h.norm(dim=1, keepdim=True) + eps)
    return h.view(B, N, bins)

class MaskFeatViT(nn.Module):
    """MaskFeat: predict hand-crafted features (HOG-like) at masked patches."""
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=384, depth=6, heads=6, mlp_ratio=4.0, mask_ratio=0.6, feat_bins=8):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.encoder = TransformerEncoder(embed_dim, depth, heads, mlp_ratio)
        self.patch_size = patch_size
        self.feat_bins = feat_bins
        self.head = nn.Linear(embed_dim, feat_bins)

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
        # build patch pixel targets -> HOG-like
        ps = self.patch_size
        B, C, H, W = imgs.shape
        t = imgs.unfold(2, ps, ps).unfold(3, ps, ps).permute(0,2,3,1,4,5).contiguous().view(B, -1, C*ps*ps)
        feats = simple_hog(t, bins=self.feat_bins)
        # encoder on visible patches
        x = self.patch(imgs) + self.pos
        ids_keep, mask = self.random_mask(x)
        B, N, D = x.shape
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1,1,D))
        h = self.encoder(x_vis)
        out = torch.zeros(B, N, D, device=imgs.device)
        out.scatter_(1, ids_keep.unsqueeze(-1).repeat(1,1,D), h)
        pred = self.head(out)
        return pred, feats, mask

    def loss(self, outputs):
        pred, feats, mask = outputs
        # MSE on normalized features
        loss = (pred - feats).pow(2).mean(dim=-1)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss
