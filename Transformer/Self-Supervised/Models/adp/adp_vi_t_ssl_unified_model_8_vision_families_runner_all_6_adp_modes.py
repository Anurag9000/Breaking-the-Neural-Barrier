# ================================================
# File: adp_vit_ssl.py
# Purpose: Single-model Adaptive ViT for Self-Supervised Vision
#          supporting ALL 6 ADP modes and 8 SSL families:
#          {MAE, CAE, SimCLR, BarlowTwins, VICReg, SimSiam, MaskFeat, BEiT}
#
# Notes (Single-model policy):
# - No momentum encoders, no EMA/teacher, no dual-tower during training.
# - SimCLR uses in-batch negatives from two views of the SAME encoder.
# - SimSiam uses predictor head + stop-grad on target branch.
# - BEiT uses a fixed, non-learned tokenizer proxy (k-means codes simulated)
#   to avoid a second model at train-time. Replace with offline tokenizer codes
#   for real training.
# ================================================

from __future__ import annotations
import math, time
from dataclasses import dataclass, asdict
from typing import Optional, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Configs
# ---------------------------

@dataclass
class TrainConfig:
    lr: float = 3e-4
    weight_decay: float = 0.05
    batch_size: int = 128
    patience: int = 50
    max_epochs: int = 10_000_000
    clip_grad_norm: float = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'

@dataclass
class SearchConfig:
    mode: str = 'width_to_depth'  # {'width_to_depth','depth_to_width','alt_depth','alt_width','depth_only','width_only'}
    trials_width: int = 4
    trials_depth: int = 4
    ex_embed: int = 128
    ex_mlp: int = 256
    ex_heads: int = 1
    max_embed: int = 1024
    max_mlp: int = 4096
    max_heads: int = 16
    max_layers: int = 48

@dataclass
class ArchInit:
    img_size: int = 64
    patch: int = 8
    in_chans: int = 3
    embed_dim: int = 384
    depth: int = 8
    num_heads: int = 6
    mlp_ratio: float = 4.0
    drop: float = 0.0
    drop_path: float = 0.0

@dataclass
class ObjectiveConfig:
    objective: str = 'mae'  # {'mae','cae','simclr','barlow','vicreg','simsiam','maskfeat','beit'}
    proj_dim: int = 256            # projection head size for contrastive/non-contrastive
    temperature: float = 0.2       # SimCLR
    barlow_lambda: float = 0.0051
    vicreg_inv: float = 25.0
    vicreg_var: float = 1.0
    vicreg_cov: float = 1.0
    mae_mask_ratio: float = 0.75
    maskfeat_mask_ratio: float = 0.6
    beit_codebook_size: int = 8192

# ---------------------------
# ViT building blocks
# ---------------------------

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.np = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)            # (B, D, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
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
        x = self.fc1(x); x = self.act(x); x = self.drop(x)
        x = self.fc2(x); x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, drop=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=drop, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, mlp_ratio, drop)
    def forward(self, x, attn_mask=None):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False, attn_mask=None)
        x = x + h
        h = self.norm2(x)
        h = self.mlp(h)
        x = x + h
        return x

class AdaptiveViT(nn.Module):
    def __init__(self, arch: ArchInit):
        super().__init__()
        self.img_size = arch.img_size
        self.patch = arch.patch
        self.embed_dim = arch.embed_dim
        self.depth = arch.depth
        self.num_heads = arch.num_heads
        self.mlp_ratio = arch.mlp_ratio
        self.drop = arch.drop

        self.patch_embed = PatchEmbed(arch.img_size, arch.patch, arch.in_chans, arch.embed_dim)
        self.cls = nn.Parameter(torch.zeros(1, 1, arch.embed_dim))
        self.pos = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.np, arch.embed_dim))
        self.blocks = nn.ModuleList([Block(arch.embed_dim, arch.num_heads, arch.mlp_ratio, arch.drop) for _ in range(arch.depth)])
        self.norm = nn.LayerNorm(arch.embed_dim)

        # Heads for SSL tasks
        self.decoder_mae = nn.Linear(arch.embed_dim, arch.patch * arch.patch * 3)  # per-patch RGB reconstruction
        self.proj = nn.Sequential(nn.Linear(arch.embed_dim, arch.embed_dim), nn.GELU(), nn.Linear(arch.embed_dim, arch.embed_dim))
        self.head = nn.Linear(arch.embed_dim, arch.embed_dim, bias=False)  # generic head used by contrastive
        self.predictor = nn.Sequential(nn.Linear(arch.embed_dim, arch.embed_dim//2), nn.GELU(), nn.Linear(arch.embed_dim//2, arch.embed_dim))  # SimSiam predictor

        self.init_parameters()

    def init_parameters(self):
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ------ core encoder ------
    def encode(self, x):  # x: (B,3,H,W)
        B = x.size(0)
        x = self.patch_embed(x)               # (B,N,D)
        cls = self.cls.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos[:, :x.size(1), :]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x                              # (B,1+N,D)

    # ------ ADP mechanics ------
    def snapshot(self) -> Dict:
        return {
            'state_dict': {k: v.detach().cpu() for k, v in self.state_dict().items()},
            'arch': {
                'embed_dim': self.embed_dim, 'depth': self.depth, 'num_heads': self.num_heads, 'mlp_ratio': self.mlp_ratio
            }
        }

    def restore(self, snap: Dict):
        arch = snap['arch']
        self._rebuild(int(arch['embed_dim']), int(arch['depth']), int(arch['num_heads']), float(arch['mlp_ratio']))
        self.load_state_dict(snap['state_dict'], strict=True)

    def _rebuild(self, embed_dim, depth, num_heads, mlp_ratio):
        old_sd = {k: v.detach().cpu() for k, v in self.state_dict().items()}
        device = next(self.parameters()).device
        self.embed_dim, self.depth, self.num_heads, self.mlp_ratio = embed_dim, depth, num_heads, mlp_ratio
        self.patch_embed = PatchEmbed(self.img_size, self.patch, 3, embed_dim).to(device)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim).to(device))
        self.pos = nn.Parameter(torch.zeros(1, 1 + (self.img_size//self.patch)**2, embed_dim).to(device))
        self.blocks = nn.ModuleList([Block(embed_dim, num_heads, mlp_ratio, self.drop).to(device) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim).to(device)
        self.decoder_mae = nn.Linear(embed_dim, self.patch * self.patch * 3).to(device)
        self.proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.GELU(), nn.Linear(embed_dim, embed_dim)).to(device)
        self.head = nn.Linear(embed_dim, embed_dim, bias=False).to(device)
        self.predictor = nn.Sequential(nn.Linear(embed_dim, embed_dim//2), nn.GELU(), nn.Linear(embed_dim//2, embed_dim)).to(device)
        new_sd = self.state_dict()
        for k in new_sd.keys():
            if k in old_sd:
                old, new = old_sd[k], new_sd[k]
                slices = tuple(min(a, b) for a, b in zip(old.shape, new.shape))
                idx_old = tuple(slice(0, s) for s in slices)
                idx_new = tuple(slice(0, s) for s in slices)
                with torch.no_grad(): new[idx_new] = old[idx_old]
        self.load_state_dict(new_sd, strict=False)

    def widen(self, sc: SearchConfig):
        d = min(self.embed_dim + sc.ex_embed, sc.max_embed)
        heads = min(self.num_heads + sc.ex_heads, sc.max_heads)
        if d % heads != 0:
            d = (d // heads) * heads
        mlp_ratio = min((self.mlp_ratio * self.embed_dim + sc.ex_mlp) / d, sc.max_mlp / d)
        self._rebuild(d, self.depth, heads, mlp_ratio)

    def append_layer(self, sc: SearchConfig):
        depth = min(self.depth + 1, sc.max_layers)
        self._rebuild(self.embed_dim, depth, self.num_heads, self.mlp_ratio)

    def total_params(self):
        return sum(p.numel() for p in self.parameters())

    # -----------------
    # SSL losses
    # -----------------

    # MAE: reconstruct masked patches (pixel space)
    def loss_mae(self, batch, oc: ObjectiveConfig):
        imgs = batch['img']  # (B,3,H,W)
        B = imgs.size(0)
        # patchify
        ph = pw = self.patch
        unfold = nn.Unfold(kernel_size=(ph,pw), stride=(ph,pw))
        patches = unfold(imgs).transpose(1,2)  # (B,N,3*ph*pw)
        N = patches.size(1)
        # mask
        num_mask = int(oc.mae_mask_ratio * N)
        perm = torch.rand(B, N, device=imgs.device).argsort(dim=1)
        mask_idx = perm[:, :num_mask]
        keep_idx = perm[:, num_mask:]
        # encode only visible patches
        x = self.patch_embed(imgs)
        cls = self.cls.expand(B, -1, -1)
        pos = self.pos[:, :1+N]
        x = torch.cat([cls, x], dim=1) + pos
        for blk in self.blocks: x = blk(x)
        x = self.norm(x)
        # gather visible tokens (drop CLS) then decode to pixel patches
        tokens = x[:,1:,:]
        gather = tokens.gather(1, keep_idx.unsqueeze(-1).expand(-1,-1,tokens.size(-1)))
        rec = self.decoder_mae(gather)  # (B, N_keep, P)
        # target masked patches
        target = patches.gather(1, mask_idx.unsqueeze(-1).expand(-1,-1,patches.size(-1)))
        loss = F.mse_loss(rec, target)
        return loss, {'mae_mse': loss.item()}

    # CAE: reconstruct features of an internal projection for ALL patches
    def loss_cae(self, batch, oc: ObjectiveConfig):
        imgs = batch['img']
        h = self.encode(imgs)
        z = h[:,1:,:]                 # (B,N,D)
        feat = self.proj(z).detach()  # target features (stop-grad single model)
        pred = self.proj(z)
        loss = F.mse_loss(pred, feat)
        return loss, {'cae_mse': loss.item()}

    # SimCLR: two views -> projection -> NT-Xent
    def loss_simclr(self, batch, oc: ObjectiveConfig):
        xa, xb = batch['img1'], batch['img2']
        za = self.head(self.encode(xa)[:,0])
        zb = self.head(self.encode(xb)[:,0])
        za = F.normalize(za, dim=-1); zb = F.normalize(zb, dim=-1)
        logits = (za @ zb.t()) / oc.temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(1) == labels).float().mean().item()
        return loss, {'ntxent_acc': acc}

    # Barlow Twins
    def loss_barlow(self, batch, oc: ObjectiveConfig):
        xa, xb = batch['img1'], batch['img2']
        za = self.head(self.encode(xa)[:,0])
        zb = self.head(self.encode(xb)[:,0])
        za = (za - za.mean(0)) / (za.std(0) + 1e-9)
        zb = (zb - zb.mean(0)) / (zb.std(0) + 1e-9)
        N = za.size(0)
        c = (za.T @ zb) / (N - 1)
        on = torch.diagonal(c).add_(-1).pow_(2).sum()
        off = (c - torch.diag(torch.diag(c))).pow_(2).sum()
        loss = on + oc.barlow_lambda * off
        return loss, {'barlow_on': on.item(), 'barlow_off': off.item()}

    # VICReg
    def loss_vicreg(self, batch, oc: ObjectiveConfig):
        xa, xb = batch['img1'], batch['img2']
        za = self.head(self.encode(xa)[:,0])
        zb = self.head(self.encode(xb)[:,0])
        inv = F.mse_loss(za, zb)
        def variance(z):
            eps = 1e-4
            std = torch.sqrt(z.var(dim=0) + eps)
            return torch.mean(F.relu(1 - std))
        def covariance(z):
            z = z - z.mean(0, keepdim=True)
            N, D = z.shape
            cov = (z.t() @ z) / (N - 1)
            off_diag = cov - torch.diag(torch.diag(cov))
            return (off_diag ** 2).sum() / D
        var = variance(za) + variance(zb)
        cov = covariance(za) + covariance(zb)
        loss = oc.vicreg_inv*inv + oc.vicreg_var*var + oc.vicreg_cov*cov
        return loss, {'vic_inv': inv.item(), 'vic_var': var.item(), 'vic_cov': cov.item()}

    # SimSiam (stop-grad target)
    def loss_simsiam(self, batch, oc: ObjectiveConfig):
        xa, xb = batch['img1'], batch['img2']
        za = self.head(self.encode(xa)[:,0])
        zb = self.head(self.encode(xb)[:,0])
        pa = self.predictor(za)
        pb = self.predictor(zb)
        def D(p, z):
            z = z.detach()
            p = F.normalize(p, dim=-1); z = F.normalize(z, dim=-1)
            return - (p * z).sum(dim=-1).mean()
        loss = D(pa, zb) * 0.5 + D(pb, za) * 0.5
        return loss, {'simsiam': loss.item()}

    # MaskFeat: predict per-patch features of masked subset (use simple DCT proxy)
    def loss_maskfeat(self, batch, oc: ObjectiveConfig):
        imgs = batch['img']
        B = imgs.size(0)
        ph = pw = self.patch
        unfold = nn.Unfold(kernel_size=(ph,pw), stride=(ph,pw))
        patches = unfold(imgs).transpose(1,2)  # (B,N,3*ph*pw)
        N = patches.size(1)
        num_mask = int(oc.maskfeat_mask_ratio * N)
        perm = torch.rand(B, N, device=imgs.device).argsort(1)
        mask_idx = perm[:, :num_mask]
        keep_idx = perm[:, num_mask:]
        # fixed random projection as "feature"
        Pdim = min(256, patches.size(-1))
        proj = getattr(self, 'maskfeat_proj', None)
        if proj is None:
            self.maskfeat_proj = nn.Linear(patches.size(-1), Pdim, bias=False).to(imgs.device)
            with torch.no_grad(): self.maskfeat_proj.weight.data.normal_(0, 0.02)
            proj = self.maskfeat_proj
        feats = proj(patches)  # (B,N,Pdim)
        # encode visible tokens and predict masked features
        x = self.encode(imgs)[:,1:,:]
        gather = x.gather(1, keep_idx.unsqueeze(-1).expand(-1,-1,x.size(-1)))
        pred = self.head(gather)
        target = feats.gather(1, mask_idx.unsqueeze(-1).expand(-1,-1,feats.size(-1)))
        loss = F.mse_loss(pred, target)
        return loss, {'maskfeat_mse': loss.item()}

    # BEiT: discrete token prediction for masked patches using a fixed codebook proxy
    def loss_beit(self, batch, oc: ObjectiveConfig):
        imgs = batch['img']
        B = imgs.size(0)
        ph = pw = self.patch
        unfold = nn.Unfold(kernel_size=(ph,pw), stride=(ph,pw))
        patches = unfold(imgs).transpose(1,2)
        N = patches.size(1)
        num_mask = int(0.4 * N)
        perm = torch.rand(B, N, device=imgs.device).argsort(1)
        mask_idx = perm[:, :num_mask]
        keep_idx = perm[:, num_mask:]
        # codebook proxy: fixed random projection -> argmax bucket
        codebook = getattr(self, 'beit_codebook', None)
        if codebook is None:
            K = oc.beit_codebook_size
            D = patches.size(-1)
            self.beit_codebook = nn.Parameter(torch.randn(K, D) * 0.02, requires_grad=False).to(imgs.device)
            codebook = self.beit_codebook
        with torch.no_grad():
            sim = patches @ codebook.t()  # (B,N,K)
            target_ids = sim.argmax(-1)   # (B,N)
        # encode visible tokens and classify masked to K codes
        x = self.encode(imgs)[:,1:,:]
        gather = x.gather(1, keep_idx.unsqueeze(-1).expand(-1,-1,x.size(-1)))
        classifier = getattr(self, 'beit_cls', None)
        if classifier is None:
            self.beit_cls = nn.Linear(self.embed_dim, oc.beit_codebook_size).to(imgs.device)
            classifier = self.beit_cls
        logits = classifier(gather)
        tgt = target_ids.gather(1, mask_idx)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1))
        return loss, {'beit_ce': loss.item()}

    # dispatch
    def objective_step(self, batch, oc: ObjectiveConfig):
        o = oc.objective.lower()
        if o == 'mae': return self.loss_mae(batch, oc)
        if o == 'cae': return self.loss_cae(batch, oc)
        if o == 'simclr': return self.loss_simclr(batch, oc)
        if o == 'barlow': return self.loss_barlow(batch, oc)
        if o == 'vicreg': return self.loss_vicreg(batch, oc)
        if o == 'simsiam': return self.loss_simsiam(batch, oc)
        if o == 'maskfeat': return self.loss_maskfeat(batch, oc)
        if o == 'beit': return self.loss_beit(batch, oc)
        raise ValueError(f'Unknown objective {o}')

# ---------------------------
# Trainer with 6 ADP modes
# ---------------------------

class ADPTrainer:
    def __init__(self, model: AdaptiveViT, tc: TrainConfig, sc: SearchConfig, oc: ObjectiveConfig):
        self.model = model.to(tc.device)
        self.tc, self.sc, self.oc = tc, sc, oc
        self.opt = torch.optim.AdamW(self.model.parameters(), lr=tc.lr, weight_decay=tc.weight_decay)
        self.best_val = float('inf')
        self.best_snap = None
        self.global_epoch = 0

    def run_epoch(self, loader, train=True):
        self.model.train(train)
        tot, n = 0.0, 0
        meters: Dict[str, float] = {}
        for batch in loader:
            batch = {k: (v.to(self.tc.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if train: self.opt.zero_grad(set_to_none=True)
            loss, m = self.model.objective_step(batch, self.oc)
            if train:
                loss.backward()
                if self.tc.clip_grad_norm is not None:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.tc.clip_grad_norm)
                self.opt.step()
            tot += float(loss.item()); n += 1
            for k,v in m.items(): meters[k] = meters.get(k,0.0)+float(v)
        return tot/max(n,1), {k:v/max(n,1) for k,v in meters.items()}

    def fit_es(self, tr, va):
        bad = 0
        while self.global_epoch < self.tc.max_epochs:
            trl,_ = self.run_epoch(tr, True)
            val,mm = self.run_epoch(va, False)
            self.global_epoch += 1
            if val < self.best_val:
                self.best_val = val; self.best_snap = self.model.snapshot(); bad = 0
            else:
                bad += 1
            print({'epoch':self.global_epoch,'val':round(val,4),'best':round(self.best_val,4),'bad':bad,
                   'arch':{'embed':self.model.embed_dim,'heads':self.model.num_heads,'depth':self.model.depth,'mlp':self.model.mlp_ratio},
                   'params_m': round(self.model.total_params()/1e6,3)})
            if bad >= self.tc.patience: break
        if self.best_snap is not None:
            self.model.restore(self.best_snap)
        return self.best_val

    # 6 modes
    def search(self, tr, va):
        m = self.sc.mode
        if m == 'width_to_depth':
            return self._width_then_depth(tr, va)
        if m == 'depth_to_width':
            return self._depth_then_width(tr, va)
        if m == 'alt_depth':
            return self._alternate(tr, va, first='depth')
        if m == 'alt_width':
            return self._alternate(tr, va, first='width')
        if m == 'depth_only':
            return self._depth_only(tr, va)
        if m == 'width_only':
            return self._width_only(tr, va)
        raise ValueError('unknown adp mode')

    def _width_then_depth(self, tr, va):
        print('[ADP] Phase: Width then Depth')
        for _ in range(self.sc.trials_width):
            self.fit_es(tr, va)
            self.model.widen(self.sc)
        for _ in range(self.sc.trials_depth):
            self.fit_es(tr, va)
            self.model.append_layer(self.sc)
        return self.best_val

    def _depth_then_width(self, tr, va):
        print('[ADP] Phase: Depth then Width')
        for _ in range(self.sc.trials_depth):
            self.fit_es(tr, va)
            self.model.append_layer(self.sc)
        for _ in range(self.sc.trials_width):
            self.fit_es(tr, va)
            self.model.widen(self.sc)
        return self.best_val

    def _alternate(self, tr, va, first='depth'):
        order = ['depth','width'] if first=='depth' else ['width','depth']
        td = tw = 0
        while td < self.sc.trials_depth or tw < self.sc.trials_width:
            for ph in order:
                if ph=='depth' and td < self.sc.trials_depth:
                    self.fit_es(tr, va); self.model.append_layer(self.sc); td += 1
                if ph=='width' and tw < self.sc.trials_width:
                    self.fit_es(tr, va); self.model.widen(self.sc); tw += 1
                if td>=self.sc.trials_depth and tw>=self.sc.trials_width: break
        return self.best_val

    def _depth_only(self, tr, va):
        for _ in range(self.sc.trials_depth):
            self.fit_es(tr, va); self.model.append_layer(self.sc)
        return self.best_val

    def _width_only(self, tr, va):
        for _ in range(self.sc.trials_width):
            self.fit_es(tr, va); self.model.widen(self.sc)
        return self.best_val

# ================================================
# File: run_adp_vit_ssl.py
# Purpose: Unified runner for 8 vision SSL objectives & 6 ADP modes.
# ================================================

import argparse
from typing import Tuple

class ToyVisionDataset(torch.utils.data.Dataset):
    """Tiny synthetic dataset with two augmented views for SSL.
    Replace with CIFAR/ImageNet + real augmentations for practical runs."""
    def __init__(self, n=8192, img_size=64):
        super().__init__()
        g = torch.Generator().manual_seed(42)
        self.img = torch.rand(n, 3, img_size, img_size, generator=g)
        self.n = n
        self.img_size = img_size
    def _aug(self, x):
        # light toy augmentations
        noise = (torch.rand_like(x) - 0.5) * 0.1
        x = torch.clamp(x + noise, 0, 1)
        return x
    def __getitem__(self, idx):
        x = self.img[idx]
        sample = {
            'img': x,
            'img1': self._aug(x.clone()),
            'img2': self._aug(x.clone()),
        }
        return sample
    def __len__(self):
        return self.n


def build_loaders(args, oc: ObjectiveConfig) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    ds = ToyVisionDataset(n=args.train_size, img_size=args.img_size)
    n_val = max(64, int(0.1 * len(ds)))
    n_tr = len(ds) - n_val
    tr, va = torch.utils.data.random_split(ds, [n_tr, n_val], generator=torch.Generator().manual_seed(42))
    tr_loader = torch.utils.data.DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=0)
    va_loader = torch.utils.data.DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=0)
    return tr_loader, va_loader


def main():
    p = argparse.ArgumentParser()
    # Architecture
    p.add_argument('--img_size', type=int, default=64)
    p.add_argument('--patch', type=int, default=8)
    p.add_argument('--embed_dim', type=int, default=384)
    p.add_argument('--depth', type=int, default=8)
    p.add_argument('--num_heads', type=int, default=6)
    p.add_argument('--mlp_ratio', type=float, default=4.0)

    # Objective
    p.add_argument('--objective', type=str, default='mae', choices=['mae','cae','simclr','barlow','vicreg','simsiam','maskfeat','beit'])

    # Training
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--weight_decay', type=float, default=0.05)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--patience', type=int, default=50)
    p.add_argument('--max_epochs', type=int, default=10_000_000)

    # ADP Search
    p.add_argument('--adp_mode', type=str, default='width_to_depth', choices=['width_to_depth','depth_to_width','alt_depth','alt_width','depth_only','width_only'])
    p.add_argument('--trials_width', type=int, default=2)
    p.add_argument('--trials_depth', type=int, default=2)
    p.add_argument('--ex_embed', type=int, default=128)
    p.add_argument('--ex_mlp', type=int, default=256)
    p.add_argument('--ex_heads', type=int, default=1)
    p.add_argument('--max_embed', type=int, default=1024)
    p.add_argument('--max_mlp', type=int, default=4096)
    p.add_argument('--max_heads', type=int, default=16)
    p.add_argument('--max_layers', type=int, default=48)

    # Data
    p.add_argument('--train_size', type=int, default=8192)

    args = p.parse_args()

    arch = ArchInit(img_size=args.img_size, patch=args.patch, embed_dim=args.embed_dim,
                    depth=args.depth, num_heads=args.num_heads, mlp_ratio=args.mlp_ratio)
    oc = ObjectiveConfig(objective=args.objective)
    tc = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, batch_size=args.batch_size,
                     patience=args.patience, max_epochs=args.max_epochs)
    sc = SearchConfig(mode=args.adp_mode, trials_width=args.trials_width, trials_depth=args.trials_depth,
                      ex_embed=args.ex_embed, ex_mlp=args.ex_mlp, ex_heads=args.ex_heads,
                      max_embed=args.max_embed, max_mlp=args.max_mlp, max_heads=args.max_heads, max_layers=args.max_layers)

    print('[ARCH]', asdict(arch))
    print('[OBJ ]', asdict(oc))
    print('[ADP ]', asdict(sc))

    model = AdaptiveViT(arch)
    trainer = ADPTrainer(model, tc, sc, oc)
    tr, va = build_loaders(args, oc)
    best = trainer.search(tr, va)
    print('[BEST_VAL]', best)

if __name__ == '__main__':
    main()
