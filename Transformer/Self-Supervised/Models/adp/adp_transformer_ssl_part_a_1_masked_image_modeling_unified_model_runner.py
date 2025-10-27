# =============================
# File: adp_transformer_mim.py  (MODEL)
# Single-model, self-supervised Transformer with 6 ADP search policies
# Part A(1): Masked Image Modeling (MAE / SimMIM / MaskFeat / CAE style)
# =============================

import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------
# Utility: overlap copy for resizing Linear/Embedding/LayerNorm
# -----------------

def _overlap_copy_param(dst: torch.nn.Parameter, src: torch.nn.Parameter):
    with torch.no_grad():
        sd = dst.data
        ss = src.data
        # Match trailing dims first
        slices = tuple(slice(0, min(sd.size(i), ss.size(i))) for i in range(sd.dim()))
        sd[...] = 0
        sd[slices] = ss[slices]


def _resize_linear(module: nn.Linear, in_features: int, out_features: int) -> nn.Linear:
    new = nn.Linear(in_features, out_features, bias=module.bias is not None)
    _overlap_copy_param(new.weight, module.weight)
    if module.bias is not None:
        _overlap_copy_param(new.bias, module.bias)
    return new


def _resize_layernorm(ln: nn.LayerNorm, normalized_shape: int) -> nn.LayerNorm:
    new = nn.LayerNorm(normalized_shape, eps=ln.eps, elementwise_affine=ln.elementwise_affine)
    if ln.elementwise_affine:
        _overlap_copy_param(new.weight, ln.weight)
        _overlap_copy_param(new.bias, ln.bias)
    return new


def _resize_embedding(emb: nn.Embedding, num_embeddings: int, embedding_dim: int) -> nn.Embedding:
    new = nn.Embedding(num_embeddings, embedding_dim)
    _overlap_copy_param(new.weight, emb.weight)
    return new

# -----------------
# Patchify / Unpatchify helpers (vision)
# -----------------

def patchify(imgs: torch.Tensor, patch_size: int) -> torch.Tensor:
    """ imgs: (B, C, H, W) -> patches: (B, N, C*P*P) """
    B, C, H, W = imgs.shape
    assert H % patch_size == 0 and W % patch_size == 0
    h = H // patch_size
    w = W // patch_size
    x = imgs.reshape(B, C, h, patch_size, w, patch_size)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(B, h*w, C*patch_size*patch_size)
    return x


def unpatchify(patches: torch.Tensor, patch_size: int, img_hw: Tuple[int, int], C: int) -> torch.Tensor:
    """ patches: (B, N, C*P*P) -> imgs: (B, C, H, W) """
    B, N, PP = patches.shape
    H, W = img_hw
    h = H // patch_size
    w = W // patch_size
    x = patches.reshape(B, h, w, C, patch_size, patch_size).permute(0, 3, 1, 4, 2, 5)
    x = x.reshape(B, C, H, W)
    return x

# -----------------
# Transformer blocks
# -----------------

class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.fc1 = nn.Linear(dim, hidden_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_dim, dim)
    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))

class TransformerBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim*mlp_ratio))
    def forward(self, x):
        h = self.ln1(x)
        a,_ = self.attn(h, h, h)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

# -----------------
# Adaptive SSL ViT (vision-only for Part A1)
# -----------------

@dataclass
class ADPConfig:
    image_size: int = 224
    patch_size: int = 16
    in_chans: int = 3
    dim: int = 256          # width (hidden size)
    depth: int = 6          # number of encoder blocks
    heads: int = 8          # attention heads
    mlp_ratio: float = 4.0
    mask_ratio: float = 0.75
    ssl_variant: str = "mae"   # {mae, simmim, maskfeat, cae}

    # ADP search caps
    max_dim: int = 1024
    max_depth: int = 24

    # decoder for MAE/CAE
    dec_dim: int = 256
    dec_depth: int = 2

class AdaptiveTransformerMIM(nn.Module):
    def __init__(self, cfg: ADPConfig):
        super().__init__()
        self.cfg = cfg

        self.num_patches = (cfg.image_size // cfg.patch_size) ** 2
        self.patch_embed = nn.Linear(cfg.in_chans * cfg.patch_size * cfg.patch_size, cfg.dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, cfg.dim))

        self.blocks = nn.ModuleList([
            TransformerBlock(cfg.dim, cfg.heads, cfg.mlp_ratio) for _ in range(cfg.depth)
        ])
        self.norm = nn.LayerNorm(cfg.dim)

        # Lightweight decoder (shared)
        self.use_decoder = cfg.ssl_variant in {"mae", "cae"}
        if self.use_decoder:
            self.dec_proj = nn.Linear(cfg.dim, cfg.dec_dim)
            self.dec_blocks = nn.ModuleList([
                TransformerBlock(cfg.dec_dim, max(1, cfg.dec_dim // 64), 4.0) for _ in range(cfg.dec_depth)
            ])
            self.dec_norm = nn.LayerNorm(cfg.dec_dim)
            self.dec_head = nn.Linear(cfg.dec_dim, cfg.in_chans * cfg.patch_size * cfg.patch_size)

        # For SimMIM: simple linear head from encoder tokens to pixels
        self.pix_head = nn.Linear(cfg.dim, cfg.in_chans * cfg.patch_size * cfg.patch_size)

        # For MaskFeat: fixed target extraction (HOG-like with convs) done outside; here just linear
        self.feat_head = nn.Linear(cfg.dim, 192)  # 192 dummy feature dims (e.g., HOG bins)

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    # -----------------
    # ADP inspectors
    # -----------------
    @property
    def width(self) -> int:
        return self.cfg.dim

    @property
    def total_neurons(self) -> int:
        # Rough proxy: model dim * depth (attention+MLP) + decoder if present
        core = self.cfg.dim * (self.cfg.depth * 2)
        dec = 0
        if self.use_decoder:
            dec = self.cfg.dec_dim * (self.cfg.dec_depth * 2)
        return core + dec

    # -----------------
    # Width expansion (increase dim)
    # -----------------
    def widen(self, ex_k: int):
        new_dim = min(self.cfg.dim + ex_k, self.cfg.max_dim)
        if new_dim == self.cfg.dim:
            return False
        # resize patch_embed
        self.patch_embed = _resize_linear(self.patch_embed, self.patch_embed.in_features, new_dim)
        # pos embed
        new_pos = nn.Parameter(torch.zeros(1, self.num_patches, new_dim))
        _overlap_copy_param(new_pos, self.pos_embed)
        self.pos_embed = new_pos
        # blocks
        new_blocks = []
        for blk in self.blocks:
            nb = TransformerBlock(new_dim, max(1, (new_dim // 32)), blk.mlp.fc1.out_features // blk.mlp.fc2.out_features if False else 4.0)
            # copy ln1
            nb.ln1 = _resize_layernorm(blk.ln1, new_dim)
            # attn q,k,v,out are inside nn.MultiheadAttention -> rebuild with new dim
            nb.attn = nn.MultiheadAttention(new_dim, max(1, (new_dim // 32)), batch_first=True)
            # ln2
            nb.ln2 = _resize_layernorm(blk.ln2, new_dim)
            # mlp
            nb.mlp = MLP(new_dim, int(new_dim * 4.0))
            new_blocks.append(nb)
        self.blocks = nn.ModuleList(new_blocks)
        self.norm = _resize_layernorm(self.norm, new_dim)
        # heads
        self.pix_head = _resize_linear(self.pix_head, new_dim, self.pix_head.out_features)
        self.feat_head = _resize_linear(self.feat_head, new_dim, self.feat_head.out_features)
        if self.use_decoder:
            self.dec_proj = _resize_linear(self.dec_proj, new_dim, self.dec_proj.out_features)
        self.cfg.dim = new_dim
        return True

    # -----------------
    # Depth expansion (append encoder block)
    # -----------------
    def deepen(self):
        if self.cfg.depth >= self.cfg.max_depth:
            return False
        blk = TransformerBlock(self.cfg.dim, max(1, (self.cfg.dim // 32)), 4.0)
        self.blocks.append(blk)
        self.cfg.depth += 1
        return True

    # -----------------
    # Forward + SSL heads
    # -----------------
    def encode_tokens(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W)
        patches = patchify(x, self.cfg.patch_size)                     # (B, N, P)
        tokens = self.patch_embed(patches) + self.pos_embed            # (B, N, D)
        for blk in self.blocks:
            tokens = blk(tokens)
        tokens = self.norm(tokens)
        return tokens, patches

    def forward_mae(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        # mask: (B, N) 1=masked
        tokens, patches = self.encode_tokens(x)
        # keep only visible tokens for decoder bottleneck (like MAE)
        visible = tokens[~mask].reshape(x.size(0), -1, tokens.size(-1))
        z = self.dec_proj(visible)
        for blk in self.dec_blocks:
            z = blk(z)
        z = self.dec_norm(z)
        pred_vis = self.dec_head(z)  # predictions for visible positions only
        # Scatter back to full sequence shape (predict only masked tokens -> here simplifying: predict all then compute masked loss)
        # Simple variant: predict all from encoder tokens via pix_head and only compute loss on masked positions
        logits = self.pix_head(tokens)
        target = patches
        loss = F.mse_loss(logits[mask], target[mask])
        return loss, {"pred": logits, "target": target}

    def forward_simmim(self, x: torch.Tensor, mask: torch.Tensor):
        tokens, patches = self.encode_tokens(x)
        pred = self.pix_head(tokens)
        loss = F.l1_loss(pred[mask], patches[mask])
        return loss, {"pred": pred, "target": patches}

    def forward_maskfeat(self, x: torch.Tensor, mask: torch.Tensor):
        # Simple hand-crafted target features: average-pooled RGB + Sobel mags (no extra nets)
        with torch.no_grad():
            B, C, H, W = x.shape
            gx = F.conv2d(x, torch.tensor([[[[-1,0,1],[-2,0,2],[-1,0,1]]]]*C, device=x.device, dtype=x.dtype), padding=1, groups=C)
            gy = F.conv2d(x, torch.tensor([[[[-1,-2,-1],[0,0,0],[1,2,1]]]]*C, device=x.device, dtype=x.dtype), padding=1, groups=C)
            mag = torch.sqrt(gx**2 + gy**2 + 1e-6)
            feat_img = torch.cat([x, mag], dim=1)  # (B, 2C, H, W)
            feats = patchify(feat_img, self.cfg.patch_size)  # (B, N, 2C*P*P)
            # reduce to 192 via avg pool over channels to keep it simple
            feats = feats.mean(dim=-1, keepdim=True).expand(-1, -1, 192)
        tokens, _ = self.encode_tokens(x)
        pred = self.feat_head(tokens)
        loss = F.mse_loss(pred[mask], feats[mask])
        return loss, {"pred": pred}

    def forward_cae(self, x: torch.Tensor, mask: torch.Tensor):
        # Encode then decode all tokens; compute pixel MSE on masked ones
        tokens, patches = self.encode_tokens(x)
        z = self.dec_proj(tokens)
        for blk in self.dec_blocks:
            z = blk(z)
        z = self.dec_norm(z)
        pred = self.dec_head(z)
        loss = F.mse_loss(pred[mask], patches[mask])
        return loss, {"pred": pred, "target": patches}

    def forward(self, x: torch.Tensor, mask: torch.Tensor):
        if self.cfg.ssl_variant == "mae":
            return self.forward_mae(x, mask)
        if self.cfg.ssl_variant == "simmim":
            return self.forward_simmim(x, mask)
        if self.cfg.ssl_variant == "maskfeat":
            return self.forward_maskfeat(x, mask)
        if self.cfg.ssl_variant == "cae":
            return self.forward_cae(x, mask)
        raise ValueError(f"Unknown ssl_variant: {self.cfg.ssl_variant}")

# -----------------
# ADP search policies (6 variants)
# -----------------

@dataclass
class SearchCfg:
    algo: str = "wd"          # {wd,dw,alt_d,alt_w,depth_only,width_only}
    ex_k: int = 32             # width increment (d_model)
    delta: float = 1e-3        # required loss improvement
    trials_width: int = 3
    trials_depth: int = 3
    max_neurons: int = 2_000_000


def make_mask(B: int, N: int, ratio: float, device: torch.device):
    keep = int((1.0 - ratio) * N)
    mask = torch.ones(B, N, dtype=torch.bool, device=device)
    for b in range(B):
        idx = torch.randperm(N, device=device)
        mask[b, idx[:keep]] = False
    return mask


def train_inner(model: AdaptiveTransformerMIM, images: torch.Tensor, steps: int, lr: float, mask_ratio: float) -> float:
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    best = float("inf")
    for _ in range(steps):
        mask = make_mask(images.size(0), model.num_patches, mask_ratio, images.device)
        loss,_ = model(images, mask)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        best = min(best, float(loss.item()))
    return best


def adp_search(model: AdaptiveTransformerMIM, images: torch.Tensor, s: SearchCfg, lr: float) -> Dict:
    """Minimal demonstration loop over a single batch to keep the file compact.
    In your training script, plug real dataloaders and validation here.
    """
    def accept(cur_best, new_best):
        return new_best < (cur_best - s.delta)

    baseline = train_inner(model, images, steps=5, lr=lr, mask_ratio=model.cfg.mask_ratio)
    width_fails = depth_fails = 0

    if s.algo == "depth_only":
        while depth_fails < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen():
                break
            new = train_inner(model, images, 5, lr, model.cfg.mask_ratio)
            if accept(baseline, new):
                baseline = new
                depth_fails = 0
            else:
                depth_fails += 1
        return {"best": baseline}

    if s.algo == "width_only":
        while width_fails < s.trials_width and model.total_neurons < s.max_neurons:
            grew = model.widen(s.ex_k)
            if not grew:
                break
            new = train_inner(model, images, 5, lr, model.cfg.mask_ratio)
            if accept(baseline, new):
                baseline = new
                width_fails = 0
            else:
                width_fails += 1
        return {"best": baseline}

    def width_series():
        nonlocal baseline, width_fails
        width_fails = 0
        improved = False
        while width_fails < s.trials_width and model.total_neurons < s.max_neurons:
            if not model.widen(s.ex_k):
                break
            new = train_inner(model, images, 5, lr, model.cfg.mask_ratio)
            if accept(baseline, new):
                baseline = new
                improved = True
                width_fails = 0
            else:
                width_fails += 1
        return improved

    def depth_series():
        nonlocal baseline, depth_fails
        depth_fails = 0
        improved = False
        while depth_fails < s.trials_depth and model.total_neurons < s.max_neurons:
            if not model.deepen():
                break
            new = train_inner(model, images, 5, lr, model.cfg.mask_ratio)
            if accept(baseline, new):
                baseline = new
                improved = True
                depth_fails = 0
            else:
                depth_fails += 1
        return improved

    if s.algo == "wd":
        # Width → Depth
        if width_series():
            depth_series()
        return {"best": baseline}

    if s.algo == "dw":
        # Depth → Width
        if depth_series():
            width_series()
        return {"best": baseline}

    if s.algo == "alt_d":
        # alternating depth-first cycles
        while True:
            imp_d = depth_series()
            imp_w = width_series()
            if not imp_d and not imp_w:
                break
        return {"best": baseline}

    if s.algo == "alt_w":
        while True:
            imp_w = width_series()
            imp_d = depth_series()
            if not imp_w and not imp_d:
                break
        return {"best": baseline}

    raise ValueError(f"Unknown algo {s.algo}")


# =============================
# File: run_adp_transformer_mim.py (RUNNER)
# Minimal CLI to demonstrate selection of ADP algo + SSL variant for Part A1
# =============================

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    # Part selector (we are implementing only A1 in this file)
    p.add_argument("--part", default="mim", choices=["mim"], help="Self-supervised family (this file = MIM only)")
    p.add_argument("--ssl-variant", default="mae", choices=["mae","simmim","maskfeat","cae"], help="MIM variant")

    # Model/patch/resize
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--patch-size", type=int, default=16)
    p.add_argument("--dim", type=int, default=256)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--mask-ratio", type=float, default=0.75)

    # ADP
    p.add_argument("--algo", default="wd", choices=["wd","dw","alt_d","alt_w","depth_only","width_only"]) 
    p.add_argument("--ex-k", type=int, default=32)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--trials-width", type=int, default=3)
    p.add_argument("--trials-depth", type=int, default=3)

    # Train demo
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")

    args = p.parse_args()

    cfg = ADPConfig(
        image_size=args.image_size,
        patch_size=args.patch_size,
        dim=args.dim,
        depth=args.depth,
        heads=args.heads,
        mask_ratio=args.mask_ratio,
        ssl_variant=args.ssl_variant,
    )
    model = AdaptiveTransformerMIM(cfg).to(args.device)

    # Demo synthetic batch (replace with real dataloader for CIFAR/ImageNet)
    imgs = torch.randn(args.batch, 3, args.image_size, args.image_size, device=args.device)

    scfg = SearchCfg(
        algo=args.algo,
        ex_k=args.ex_k,
        delta=args.delta,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
    )

    out = adp_search(model, imgs, scfg, lr=args.lr)
    print({"best_demo_loss": out["best"], "final_dim": model.cfg.dim, "final_depth": model.cfg.depth, "neurons": model.total_neurons})
