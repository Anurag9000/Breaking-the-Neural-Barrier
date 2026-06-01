import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchEmbed(nn.Module):
    def __init__(self, img=128, patch=8, in_ch=3, dim=256):
        super().__init__(); self.grid=(img//patch, img//patch)
        self.proj = nn.Conv2d(in_ch, dim, patch, patch)
    def forward(self, x):
        x = self.proj(x); B,C,H,W = x.shape
        return x.flatten(2).transpose(1,2), H, W

class PromptableSegViT(nn.Module):
    """SAM-style inspiration but *single-model*: ViT encoder + prompt token embedding for a point prompt -> mask.
    Prompts are (x,y) in image coords, normalized to patch grid and embedded as a token concatenated to patch tokens.
    """
    def __init__(self, img=128, patch=8, dim=256, depth=6, nhead=8, mlp_ratio=4.0):
        super().__init__()
        self.patch = PatchEmbed(img, patch, 3, dim)
        self.pos = nn.Parameter(torch.zeros(1, (img//patch)*(img//patch)+1, dim))
        enc = nn.TransformerEncoderLayer(dim, nhead, int(dim*mlp_ratio), 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.prompt_mlp = nn.Sequential(nn.Linear(2, dim), nn.GELU(), nn.Linear(dim, dim))
        self.mask_head = nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, dim))
        self.dim = dim; self.img=img; self.patch_size=patch
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x, points):
        # x: (B,3,H,W), points: (B,2) in [0,1] normalized image coords (x,y)
        B,C,H,W = x.shape
        z, h, w = self.patch(x)
        # prompt token
        # convert to patch coords (0..w-1, 0..h-1)
        pw = points.clone(); pw[...,0] = pw[...,0]*(w-1); pw[...,1] = pw[...,1]*(h-1)
        p_tok = self.prompt_mlp(pw)
        p_tok = p_tok.unsqueeze(1)  # (B,1,D)
        tokens = torch.cat([p_tok, z], dim=1)
        tokens = tokens + self.pos[:, : tokens.size(1)]
        z = self.encoder(tokens)
        # use prompt token features to project patch tokens into mask
        p_feat = z[:,0:1,:]
        patch_feats = z[:,1:,:]
        mask_logits = (self.mask_head(p_feat) @ patch_feats.transpose(1,2)).view(B,1,h,w)
        mask_logits = F.interpolate(mask_logits, size=(H,W), mode='bilinear', align_corners=False)
        return mask_logits.squeeze(1)
