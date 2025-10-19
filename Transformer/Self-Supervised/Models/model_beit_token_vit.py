import torch
import torch.nn as nn
import torch.nn.functional as F
from model_mae_vit import PatchEmbed, TransformerEncoder

class SimpleCodebook(nn.Module):
    """Frozen codebook for token targets (initialize once from random patches)."""
    def __init__(self, dim=192, k=8192):
        super().__init__()
        self.register_buffer('centroids', torch.randn(k, dim))
        self.centroids = nn.Parameter(self.centroids, requires_grad=False)

    @torch.no_grad()
    def init_from_samples(self, feats, iters=10):
        # kmeans-lite init on given features (NxD)
        C = self.centroids.shape[0]
        idx = torch.randperm(feats.size(0), device=feats.device)[:C]
        self.centroids.copy_(feats[idx])
        for _ in range(iters):
            d = torch.cdist(feats, self.centroids)
            a = d.argmin(dim=1)
            for c in range(C):
                m = (a==c)
                if m.any():
                    self.centroids[c] = feats[m].mean(dim=0)

class BEiTTokenViT(nn.Module):
    """BEiT-style: predict discrete visual tokens of masked patches (no teacher; offline codebook).
    We build a simple codebook over linear patch-projections and freeze it during training.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3,
                 embed_dim=384, depth=6, heads=6, mlp_ratio=4.0,
                 mask_ratio=0.4, code_dim=192, code_k=8192):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.patch = PatchEmbed(img_size, patch_size, in_chans, embed_dim)
        self.pos = nn.Parameter(torch.zeros(1, (img_size//patch_size)**2, embed_dim))
        nn.init.trunc_normal_(self.pos, std=0.02)
        self.encoder = TransformerEncoder(embed_dim, depth, heads, mlp_ratio)
        self.proj_for_code = nn.Linear(embed_dim, code_dim)
        self.codebook = SimpleCodebook(code_dim, code_k)
        self.head = nn.Linear(embed_dim, code_k)

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

    @torch.no_grad()
    def build_targets(self, x):
        # x: (B,N,D) encoder inputs (pre-mask) -> project then assign nearest centroid
        feats = self.proj_for_code(x).reshape(-1, self.proj_for_code.out_features)
        d = torch.cdist(feats, self.codebook.centroids)
        idx = d.argmin(dim=1)
        return idx.view(x.shape[0], x.shape[1])

    def forward(self, imgs):
        x = self.patch(imgs) + self.pos
        ids_keep, mask = self.random_mask(x)
        B, N, D = x.shape
        targets = self.build_targets(x)  # (B,N)
        x_vis = torch.gather(x, 1, ids_keep.unsqueeze(-1).repeat(1,1,D))
        h = self.encoder(x_vis)
        # scatter back logits to N
        out = torch.zeros(B, N, D, device=imgs.device)
        out.scatter_(1, ids_keep.unsqueeze(-1).repeat(1,1,D), h)
        logits = self.head(out)  # (B,N,K)
        return logits, targets, mask

    def loss(self, outputs):
        logits, targets, mask = outputs
        B, N, K = logits.shape
        loss = F.cross_entropy(logits.view(B*N, K), targets.view(-1), reduction='none').view(B, N)
        loss = (loss * mask).sum() / (mask.sum() + 1e-6)
        return loss
