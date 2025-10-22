# adp_ae_spatial_core.py
# Single-model spatial/geometric SSL AEs for tasks:
#  - jigsaw_spatial (28)         : patch permutation classification (+ recon)
#  - rotation_spatial (29)       : rotation angle classification (+ recon)
#  - flip_pred (30)              : flip type classification (+ recon)
#  - scale_pred (31)             : scale-bin classification (+ recon to canonical)
#  - translate_pred (32)         : 2D offset regression (+ recon to canonical)
#  - patch_contrast (33)         : patch embedding contrastive consistency (single encoder)
#  - relative_position (34)      : relative offset class between two patches
#  - edge_completion (35)        : reconstruct edge maps in masked regions
#
# All tasks use a single encoder/decoder AE; extra heads are light MLPs in the same model.
# Author: ADP / Breaking Neural Barrier

from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple, List
import math, random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

# ----------------------------
# Config
# ----------------------------
@dataclass
class SAEConfig:
    in_channels: int = 3
    base_channels: int = 64
    depth: int = 4
    norm: str = "bn"         # 'bn'|'gn'|'ln'|'none'
    act: str = "relu"        # 'relu'|'gelu'|'silu'
    latent_dim: int = 256    # global pooled latent
    patch_grid: int = 3      # grid for patch-based tasks (>=2)
    jigsaw_classes: int = 6  # number of permutations used
    scale_bins: int = 3      # e.g., {0.9,1.0,1.1}
    trans_max_px: int = 8    # max |dx|,|dy| in pixels for translation task
    recon_loss: str = "mse"  # 'mse'|'l1'|'huber'
    huber_delta: float = 1.0
    edge_mask_ratio: float = 0.4  # fraction of image masked for edge_completion
    device: Optional[str] = None

# ----------------------------
# Small blocks
# ----------------------------
def _norm(c, kind):
    if kind=="bn": return nn.BatchNorm2d(c)
    if kind=="gn": return nn.GroupNorm(max(1, c//16), c)
    if kind=="ln": return nn.GroupNorm(1, c)
    return nn.Identity()

def _act(a): return {"relu":nn.ReLU(inplace=True),"gelu":nn.GELU(),"silu":nn.SiLU()}[a]

class ConvBNAct(nn.Module):
    def __init__(self, ci, co, cfg: SAEConfig):
        super().__init__()
        self.c = nn.Conv2d(ci, co, 3, 1, 1, bias=False)
        self.n = _norm(co, cfg.norm)
        self.a = _act(cfg.act)
    def forward(self,x): return self.a(self.n(self.c(x)))

class Down(nn.Module):
    def __init__(self, ci, co, cfg: SAEConfig):
        super().__init__()
        self.b1 = ConvBNAct(ci, co, cfg); self.b2 = ConvBNAct(co, co, cfg)
        self.p = nn.MaxPool2d(2)
    def forward(self,x): x=self.b1(x); x=self.b2(x); return self.p(x)

class Up(nn.Module):
    def __init__(self, ci, co, cfg: SAEConfig):
        super().__init__()
        self.u = nn.Upsample(scale_factor=2, mode="nearest")
        self.b1 = ConvBNAct(ci, co, cfg); self.b2 = ConvBNAct(co, co, cfg)
    def forward(self,x): x=self.u(x); x=self.b1(x); return self.b2(x)

# ----------------------------
# Encoder / Decoder
# ----------------------------
class Encoder(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        C = cfg.base_channels
        layers = [ConvBNAct(cfg.in_channels, C, cfg)]
        for _ in range(1, cfg.depth):
            layers.append(Down(C, C*2, cfg)); C*=2
        self.body = nn.Sequential(*layers)
        self.out_c = C
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(C, cfg.latent_dim)
    def forward(self, x):
        f = self.body(x)
        z = self.fc(self.gap(f).flatten(1))
        return f, z

class Decoder(nn.Module):
    def __init__(self, enc: Encoder, cfg: SAEConfig):
        super().__init__()
        C = enc.out_c
        ups=[]
        for _ in range(cfg.depth-1):
            ups.append(Up(C, C//2, cfg)); C//=2
        self.ups = nn.ModuleList(ups)
        self.head = nn.Conv2d(C, cfg.in_channels, 1)
    def forward(self, f):
        x=f
        for u in self.ups: x=u(x)
        return self.head(x)

# ----------------------------
# Heads
# ----------------------------
class ClsHead(nn.Module):
    def __init__(self, d, num):
        super().__init__()
        self.fc = nn.Linear(d, num)
    def forward(self, z): return self.fc(z)

class RegrHead(nn.Module):
    def __init__(self, d, out_dim):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.ReLU(True), nn.Linear(d, out_dim))
    def forward(self, z): return self.mlp(z)

# ----------------------------
# Patch helpers
# ----------------------------
def extract_patches(x: torch.Tensor, G: int) -> List[torch.Tensor]:
    # x: (B,C,H,W) -> list of (B,C,h,w) patches in raster order
    B,C,H,W = x.shape; h, w = H//G, W//G
    patches=[]
    for r in range(G):
        for c in range(G):
            patches.append(x[:,:,r*h:(r+1)*h, c*w:(c+1)*w])
    return patches

def make_edge_map(x: torch.Tensor) -> torch.Tensor:
    # Sobel edges per-channel, then magnitude
    sobel_x = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
    sobel_y = sobel_x.transpose(-1,-2)
    C = x.shape[1]
    gx = F.conv2d(x, sobel_x.expand(C,1,3,3), padding=1, groups=C)
    gy = F.conv2d(x, sobel_y.expand(C,1,3,3), padding=1, groups=C)
    mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
    # collapse channels by max
    return mag.max(dim=1, keepdim=True).values

# ----------------------------
# Supported algos
# ----------------------------
SUPPORTED_SPATIAL = {
    "jigsaw_spatial",      # 28
    "rotation_spatial",    # 29
    "flip_pred",           # 30
    "scale_pred",          # 31
    "translate_pred",      # 32
    "patch_contrast",      # 33 (single encoder InfoNCE-like)
    "relative_position",   # 34
    "edge_completion"      # 35
}

# ----------------------------
# Main model
# ----------------------------
class SpatialAE(nn.Module):
    def __init__(self, cfg: SAEConfig):
        super().__init__()
        self.cfg = cfg
        self.enc = Encoder(cfg)
        self.dec = Decoder(self.enc, cfg)
        D = cfg.latent_dim
        # heads
        self.head_jigsaw   = ClsHead(D, cfg.jigsaw_classes)
        self.head_rot      = ClsHead(D, 4)                 # 0/90/180/270
        self.head_flip     = ClsHead(D, 3)                 # none / hflip / vflip
        self.head_scale    = ClsHead(D, cfg.scale_bins)    # bins around 1.0
        self.head_relpos   = ClsHead(D*2, 8)               # 8-way relative offset class
        self.head_trans    = RegrHead(D, 2)                # dx, dy regression

    # utilities
    def _recon_loss(self, a, b):
        if self.cfg.recon_loss=="mse":   return F.mse_loss(a,b)
        if self.cfg.recon_loss=="l1":    return F.l1_loss(a,b)
        return F.huber_loss(a,b,delta=self.cfg.huber_delta)

    # entry
    def forward_train(self, x: torch.Tensor, algo: str) -> Dict[str, Any]:
        assert algo in SUPPORTED_SPATIAL, f"Unsupported algo {algo}"
        B,C,H,W = x.shape
        logs: Dict[str,float] = {}

        # 28) Jigsaw: sample K permutations and classify which was applied; recon permuted image
        if algo == "jigsaw_spatial":
            G = max(2, self.cfg.patch_grid)
            patches = extract_patches(x, G)  # len = G*G
            # make a small permutation set
            perms = []
            total = G*G
            perms.append(list(range(total)))          # identity
            if total >= 4:
                # swap two random indices
                a,b = 0, 1
                p = list(range(total)); p[a],p[b]=p[b],p[a]; perms.append(p)
                # rotate rows
                p = list(range(total)); p = p[G:]+p[:G]; perms.append(p)
                # rotate cols
                p = []
                for r in range(G):
                    row = list(range(r*G,(r+1)*G))
                    row = row[1:]+row[:1]
                    p += row
                perms.append(p)
            # pad to desired classes
            while len(perms) < self.cfg.jigsaw_classes:
                p = list(range(total)); random.shuffle(p); perms.append(p)
            pid = random.randrange(self.cfg.jigsaw_classes)
            order = perms[pid][:total]
            # rebuild permuted image
            h, w = H//G, W//G
            rows=[]
            for r in range(G):
                row=[]
                for c in range(G):
                    row.append(patches[order[r*G+c]])
                rows.append(torch.cat(row, dim=-1))
            x_perm = torch.cat(rows, dim=-2)
            f, z = self.enc(x_perm); logits = self.head_jigsaw(z)
            cls = F.cross_entropy(logits, torch.full((B,), pid, device=x.device, dtype=torch.long))
            rec = self._recon_loss(self.dec(f), x_perm)
            loss = cls + rec
            logs["jigsaw_ce"]=cls.item(); logs["recon"]=rec.item()
            return {"loss": loss, "logs": logs, "recon": self.dec(f)}

        # 29) Rotation: classify angle; recon canonical
        if algo == "rotation_spatial":
            k = random.choice([0,1,2,3])
            xr = torch.rot90(x, k, dims=[-2,-1])
            f, z = self.enc(xr); logits = self.head_rot(z)
            cls = F.cross_entropy(logits, torch.full((B,), k, device=x.device, dtype=torch.long))
            # decode and rotate back for recon loss
            rec = self.dec(f)
            rec_back = torch.rot90(rec, (4-k)%4, dims=[-2,-1])
            rloss = self._recon_loss(rec_back, x)
            loss = cls + rloss
            logs["rot_ce"]=cls.item(); logs["recon"]=rloss.item()
            return {"loss": loss, "logs": logs, "recon": rec_back}

        # 30) Flip: none/hflip/vflip classification; recon canonical
        if algo == "flip_pred":
            flip_id = random.choice([0,1,2])  # 0 none, 1 h, 2 v
            xf = x
            if flip_id==1: xf = torch.flip(x, dims=[-1])
            elif flip_id==2: xf = torch.flip(x, dims=[-2])
            f, z = self.enc(xf); logits = self.head_flip(z)
            cls = F.cross_entropy(logits, torch.full((B,), flip_id, device=x.device, dtype=torch.long))
            rec = self.dec(f)
            if flip_id==1: rec = torch.flip(rec, dims=[-1])
            elif flip_id==2: rec = torch.flip(rec, dims=[-2])
            rloss = self._recon_loss(rec, x)
            loss = cls + rloss
            logs["flip_ce"]=cls.item(); logs["recon"]=rloss.item()
            return {"loss": loss, "logs": logs, "recon": rec}

        # 31) Scale: classify scale bin; recon to canonical (unit) scale
        if algo == "scale_pred":
            # pick a scale from bins around 1.0
            bins = [0.9, 1.0, 1.1][:self.cfg.scale_bins]
            sid = random.randrange(len(bins)); s = bins[sid]
            # scale via affine on tensor
            xs = TF.resize(x, [int(H*s), int(W*s)])
            xs = TF.center_crop(TF.resize(xs, [H, W]), [H, W])
            f, z = self.enc(xs); logits = self.head_scale(z)
            cls = F.cross_entropy(logits, torch.full((B,), sid, device=x.device, dtype=torch.long))
            rec = self._recon_loss(self.dec(f), x)  # canonical is original x
            loss = cls + rec
            logs["scale_ce"]=cls.item(); logs["recon"]=rec.item()
            return {"loss": loss, "logs": logs, "recon": self.dec(f)}

        # 32) Translation: regress (dx,dy) applied; recon canonical
        if algo == "translate_pred":
            M = self.cfg.trans_max_px
            dx = random.randint(-M, M); dy = random.randint(-M, M)
            grid_y, grid_x = torch.meshgrid(
                torch.linspace(-1,1,H,device=x.device),
                torch.linspace(-1,1,W,device=x.device),
                indexing="ij"
            )
            # shift grid (approx): convert pixel shift to normalized coords (~2*shift/size)
            gx = grid_x - (2*dx)/(W-1); gy = grid_y - (2*dy)/(H-1)
            grid = torch.stack([gx, gy], dim=-1).unsqueeze(0).repeat(B,1,1,1)
            xt = F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=True)
            f, z = self.enc(xt); pred = self.head_trans(z)
            cls = F.smooth_l1_loss(pred, torch.tensor([dx, dy], device=x.device, dtype=pred.dtype).view(1,2).repeat(B,1))
            rec = self._recon_loss(self.dec(f), x)
            loss = cls + rec
            logs["trans_l1"]=cls.item(); logs["recon"]=rec.item()
            return {"loss": loss, "logs": logs, "recon": self.dec(f)}

        # 33) Patch-contrast: local patch embeddings of same image should be closer than mismatched (InfoNCE style)
        if algo == "patch_contrast":
            G = max(2, self.cfg.patch_grid)
            patches = extract_patches(x, G)  # list len K
            # pick two distinct patch indices per sample
            K = len(patches)
            i = random.randrange(K); j = (i + random.randint(1, K-1)) % K
            xi, xj = patches[i], patches[j]         # positives from same image
            # embed via encoder GAP fc
            _, zi = self.enc(xi); _, zj = self.enc(xj)  # (B,D)
            # normalize
            zi = F.normalize(zi, dim=1); zj = F.normalize(zj, dim=1)
            # temperature
            tau = 0.2
            # logits: pairwise dot-products (B,B)
            logits = zi @ zj.t() / tau
            targets = torch.arange(x.shape[0], device=x.device)
            ce = F.cross_entropy(logits, targets)  # in-batch negatives
            # small reconstruction to stabilize: decode xi
            fi, _ = self.enc(xi); rec = self._recon_loss(self.dec(fi), xi)
            loss = ce + 0.1 * rec
            logs["nce"]=ce.item(); logs["recon"]=rec.item()
            return {"loss": loss, "logs": logs, "recon": self.dec(fi)}

        # 34) Relative position: choose two patches; classify relative offset dir of j wrt i
        if algo == "relative_position":
            G = max(2, self.cfg.patch_grid)
            K = G*G
            # choose a pair (i,j)
            i = random.randrange(K); j = random.randrange(K)
            while j == i: j = random.randrange(K)
            ri, ci = divmod(i, G); rj, cj = divmod(j, G)
            dy, dx = rj - ri, cj - ci
            # map to 8 directions
            def dir8(dy,dx):
                if dy==0 and dx>0: return 2  # E
                if dy==0 and dx<0: return 6  # W
                if dx==0 and dy>0: return 4  # S
                if dx==0 and dy<0: return 0  # N
                if dy<0 and dx>0: return 1   # NE
                if dy<0 and dx<0: return 7   # NW
                if dy>0 and dx>0: return 3   # SE
                return 5                     # SW
            y = dir8(dy,dx)
            patches = extract_patches(x, G)
            xi, xj = patches[i], patches[j]
            _, zi = self.enc(xi); _, zj = self.enc(xj)
            zpair = torch.cat([zi, zj], dim=1)
            logits = self.head_relpos(zpair)
            ce = F.cross_entropy(logits, torch.full((B,), y, device=x.device, dtype=torch.long))
            # small recon aux
            fi, _ = self.enc(xi); rec = self._recon_loss(self.dec(fi), xi)
            loss = ce + 0.1 * rec
            logs["relpos_ce"]=ce.item(); logs["recon"]=rec.item()
            return {"loss": loss, "logs": logs, "recon": self.dec(fi)}

        # 35) Edge completion: compute Sobel edges of target; mask random block; reconstruct edges in the hole
        if algo == "edge_completion":
            # target edges
            edges = make_edge_map(x)  # (B,1,H,W)
            # mask a block on the input
            ratio = self.cfg.edge_mask_ratio
            h = int(H * math.sqrt(ratio)); w = int(W * math.sqrt(ratio))
            y0 = random.randint(0, max(0, H-h)); x0 = random.randint(0, max(0, W-w))
            mask = torch.ones(B,1,H,W,device=x.device); mask[:,:,y0:y0+h,x0:x0+w]=0.0
            xm = x * mask
            f, _ = self.enc(xm)
            pred_edges = make_edge_map(self.dec(f))
            # loss on masked region only
            wgt = 1.0 - mask
            num = ( (pred_edges - edges).abs() * wgt ).sum()
            den = wgt.sum().clamp_min(1.0)
            loss = num / den
            logs["edge_l1"]=loss.item(); logs["masked_area"]=float(w*h)/(H*W)
            # also produce a reconstruction for preview
            return {"loss": loss, "logs": logs, "recon": self.dec(f)}

        raise RuntimeError("Unhandled branch.")

def build_model(cfg: Optional[SAEConfig]=None) -> SpatialAE:
    cfg = cfg or SAEConfig()
    return SpatialAE(cfg)

if __name__ == "__main__":
    # quick smoke
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.randn(4,3,128,128).to(device)
    m = build_model(SAEConfig(device=device)).to(device)
    for algo in ["jigsaw_spatial","rotation_spatial","flip_pred","scale_pred","translate_pred","patch_contrast","relative_position","edge_completion"]:
        out = m.forward_train(x, algo)
        print(algo, "ok; loss", float(out["loss"]))
