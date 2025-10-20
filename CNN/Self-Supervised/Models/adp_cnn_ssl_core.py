
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

# ---------------------------
# CNN building blocks
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, bias: bool = True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class AdaptiveEncoder(nn.Module):
    """
    Adaptive CNN encoder with optional MaxPool after 0-based block indices in `pooling_indices`.
    GlobalAvgPool -> projection head (Linear) for SSL representation.
    """
    def __init__(self, in_ch: int, widths: List[int], pooling_indices: List[int], proj_dim: Optional[int] = None, bias: bool = True):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.pooling_indices = sorted(set(pooling_indices))
        self.bias = bias

        blocks = []
        c = in_ch
        for w in widths:
            blocks.append(ConvBNReLU(c, w, bias=bias))
            c = w
        self.convs = nn.ModuleList(blocks)
        self._pools_here = [i in self.pooling_indices for i in range(len(widths))]
        self.pool = nn.MaxPool2d(2,2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        hidden = widths[-1]
        pdim = hidden if proj_dim is None else proj_dim
        self.projector = nn.Linear(hidden, pdim, bias=True)

    @property
    def widths(self) -> List[int]:
        return [b.bn.num_features for b in self.convs]

    def forward(self, x):
        h = x
        for i, blk in enumerate(self.convs):
            h = blk(h)
            if self._pools_here[i]:
                h = self.pool(h)
        z = self.gap(h).flatten(1)
        z = self.projector(z)
        return z

    # ---- mutations ----
    def append_depth(self):
        last_c = self.convs[-1].bn.num_features
        self.convs.append(ConvBNReLU(last_c, last_c, bias=self.bias))
        self._pools_here.append(False)

    def widen_all(self, ex_k: int):
        if ex_k <= 0: return
        prev = self.in_ch
        for blk in self.convs:
            old_out = blk.bn.num_features
            new_out = old_out + ex_k
            _resize_conv2d_(blk.conv, prev, new_out)
            _resize_bn2d_(blk.bn, new_out)
            prev = new_out
        # projector input grows with last width
        _resize_linear_(self.projector, self.projector.in_features + ex_k, self.projector.out_features)

    # ---- capacity ----
    def total_neurons(self) -> int:
        enc = sum(self.widths)
        return enc + self.projector.in_features

# ---------------------------
# Resize helpers
# ---------------------------

def _overlap_copy_(dst, src):
    dims = [min(a, b) for a, b in zip(dst.shape, src.shape)]
    dst[tuple(slice(0,d) for d in dims)].copy_(src[tuple(slice(0,d) for d in dims)])

def _resize_conv2d_(conv: nn.Conv2d, in_ch: int, out_ch: int):
    old_w = conv.weight.data.clone(); old_b = conv.bias.data.clone() if conv.bias is not None else None
    k_h, k_w = conv.kernel_size; device = conv.weight.device
    conv.in_channels = in_ch; conv.out_channels = out_ch
    conv.weight = nn.Parameter(torch.empty(out_ch, in_ch, k_h, k_w, device=device))
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); _overlap_copy_(conv.weight.data, old_w)
    if conv.bias is not None:
        conv.bias = nn.Parameter(torch.zeros(out_ch, device=device))
        if old_b is not None: _overlap_copy_(conv.bias.data, old_b)

def _resize_bn2d_(bn: nn.BatchNorm2d, out_ch: int):
    device = bn.weight.device
    old_w = bn.weight.data.clone(); old_b = bn.bias.data.clone()
    old_rm = bn.running_mean.clone(); old_rv = bn.running_var.clone()
    bn.num_features = out_ch
    bn.weight = nn.Parameter(torch.ones(out_ch, device=device))
    bn.bias = nn.Parameter(torch.zeros(out_ch, device=device))
    bn.running_mean = torch.zeros(out_ch, device=device)
    bn.running_var = torch.ones(out_ch, device=device)
    _overlap_copy_(bn.weight.data, old_w); _overlap_copy_(bn.bias.data, old_b)
    _overlap_copy_(bn.running_mean, old_rm); _overlap_copy_(bn.running_var, old_rv)

def _resize_linear_(fc: nn.Linear, in_f: int, out_f: int):
    device = fc.weight.device
    old_w = fc.weight.data.clone(); old_b = fc.bias.data.clone() if fc.bias is not None else None
    fc.in_features = in_f; fc.out_features = out_f
    fc.weight = nn.Parameter(torch.empty(out_f, in_f, device=device))
    nn.init.kaiming_uniform_(fc.weight, a=math.sqrt(5)); _overlap_copy_(fc.weight.data, old_w)
    if fc.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(fc.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        fc.bias = nn.Parameter(torch.empty(out_f, device=device))
        nn.init.uniform_(fc.bias, -bound, bound)
        if old_b is not None: _overlap_copy_(fc.bias.data, old_b)

# ---------------------------
# Data (two-view SSL)
# ---------------------------

class TwoViewCIFAR10(Dataset):
    def __init__(self, root: str, train: bool, t1, t2, download: bool):
        self.base = datasets.CIFAR10(root=root, train=train, transform=None, download=download)
        self.t1, self.t2 = t1, t2
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.t1(img), self.t2(img)

def make_cifar10_ssl_loaders(data_root: str, batch_size: int, num_workers: int = 4, val_split: float = 0.1, download: bool = True, seed: int = 0, two_views: bool = True):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    aug = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2,0.2,0.2,0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])

    if not two_views:
        # fallback: duplicate a single view as both
        base_train = datasets.CIFAR10(root=data_root, train=True, transform=aug, download=download)
        class Dup(Dataset):
            def __init__(self, ds): self.ds = ds
            def __len__(self): return len(self.ds)
            def __getitem__(self, i): x, _ = self.ds[i]; return x, x
        ds_full = Dup(base_train)
    else:
        ds_full = TwoViewCIFAR10(root=data_root, train=True, t1=aug, t2=aug, download=download)

    n_val = int(len(ds_full)*val_split); n_train = len(ds_full)-n_val
    g = torch.Generator().manual_seed(seed)
    ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=g)
    ds_test = datasets.CIFAR10(root=data_root, train=False, transform=eval_tf, download=download)

    def collate(batch):
        x1 = torch.stack([b[0] for b in batch], 0)
        x2 = torch.stack([b[1] for b in batch], 0)
        return (x1, x2)

    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, dl_test

# ---------------------------
# Train + losses (SimCLR NT-Xent + optional Barlow Twins)
# ---------------------------

def barlow_twins_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
    c = (z1.T @ z2) / B
    on = torch.diagonal(c).add_(-1).pow_(2).sum()
    off = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
    return on + off

def nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """
    SimCLR NT-Xent with in-batch negatives. Single encoder used twice.
    """
    B, D = z1.shape
    z = torch.cat([z1, z2], dim=0)           # (2B, D)
    z = F.normalize(z, dim=1)
    sim = z @ z.T                             # (2B, 2B)
    mask = torch.eye(2*B, device=z.device).bool()
    sim = sim.masked_fill(mask, -9e15)        # remove diagonal

    # positives: (i <-> i+B) and (i+B <-> i)
    pos = torch.cat([torch.arange(B, 2*B), torch.arange(0, B)]).to(z.device)
    logits = sim / temperature
    labels = pos
    loss = F.cross_entropy(logits, labels)
    return loss

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    es_patience: int = 20
    grad_clip: Optional[float] = None
    lambda_ntx: float = 1.0
    lambda_barlow: float = 0.0
    temperature: float = 0.2
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SearchConfig:
    delta: float = 1e-3
    patience_width: int = 5
    patience_depth: int = 5
    ex_k: int = 8
    max_neurons: int = 1_000_000
    max_depth: int = 32
    max_width: int = 1024
    max_total_epochs: Optional[int] = None
    pooling_indices: Tuple[int, ...] = (0, 2)

class InnerTrainer:
    def __init__(self, enc: AdaptiveEncoder, cfg: TrainConfig):
        self.enc = enc; self.cfg = cfg
        self.device = cfg.device
        self.enc.to(self.device)
        self.optim = torch.optim.AdamW(self.enc.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
        self.best_val = float("inf"); self.best_state = None; self.epochs_done = 0

    def _step(self, batch, train=True):
        x1, x2 = batch
        x1 = x1.to(self.device, non_blocking=True)
        x2 = x2.to(self.device, non_blocking=True)
        if train: self.optim.zero_grad(set_to_none=True)
        z1 = self.enc(x1); z2 = self.enc(x2)
        loss = self.cfg.lambda_ntx * nt_xent_loss(z1, z2, self.cfg.temperature)
        if self.cfg.lambda_barlow > 0:
            loss = loss + self.cfg.lambda_barlow * barlow_twins_loss(z1, z2)
        if train:
            loss.backward()
            if self.cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.enc.parameters(), self.cfg.grad_clip)
            self.optim.step()
        return float(loss.item())

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.enc.eval(); tot, n = 0.0, 0
        for batch in loader:
            loss = self._step(batch, train=False)
            b = batch[0].size(0); tot += loss * b; n += b
        return tot / max(n,1)

    def fit(self, dl_train, dl_val, max_epochs=200):
        es = 0; self.best_val = float("inf"); self.best_state = None
        for _ in range(max_epochs):
            self.enc.train()
            for batch in dl_train: self._step(batch, train=True)
            val = self._eval_epoch(dl_val); self.epochs_done += 1
            if val + 1e-12 < self.best_val:
                self.best_val = val
                self.best_state = {"model": {k: v.detach().cpu().clone() for k,v in self.enc.state_dict().items()}}
                es = 0
            else:
                es += 1
            if es >= self.cfg.es_patience: break
        if self.best_state is not None: self.enc.load_state_dict(self.best_state["model"])
        return self.best_val

# ---------------------------
# Snapshot / guards
# ---------------------------

def snapshot(enc: AdaptiveEncoder) -> Dict[str, Any]:
    return {"state": {k: v.detach().cpu().clone() for k,v in enc.state_dict().items()},
            "widths": enc.widths.copy(), "pools": list(enc._pools_here)}

def restore(enc: AdaptiveEncoder, snap: Dict[str, Any]) -> None:
    curr, target = enc.widths, snap["widths"]
    if curr != target:
        new = AdaptiveEncoder(in_ch=enc.in_ch, widths=target, pooling_indices=[i for i,f in enumerate(snap["pools"]) if f],
                              proj_dim=enc.projector.out_features, bias=enc.bias)
        new.load_state_dict(snap["state"])
        enc.convs = new.convs; enc._pools_here = new._pools_here; enc.pool = new.pool; enc.gap = new.gap; enc.projector = new.projector
    else:
        enc.load_state_dict(snap["state"])

def can_widen(enc: AdaptiveEncoder, ex_k: int, scfg: SearchConfig) -> bool:
    if ex_k <= 0: return False
    projected = enc.total_neurons() + ex_k*len(enc.convs) + ex_k
    if projected > scfg.max_neurons: return False
    if any(w + ex_k > scfg.max_width for w in enc.widths): return False
    return True

def can_deepen(enc: AdaptiveEncoder, scfg: SearchConfig) -> bool:
    if len(enc.convs) + 1 > scfg.max_depth: return False
    projected = enc.total_neurons() + enc.convs[-1].bn.num_features
    return projected <= scfg.max_neurons

# ---------------------------
# Six ADP search strategies
# ---------------------------

def _train_eval_val(enc: AdaptiveEncoder, dl_train, dl_val, tcfg: TrainConfig, max_epochs: int) -> float:
    return InnerTrainer(enc, tcfg).fit(dl_train, dl_val, max_epochs=max_epochs)

def cnn_ssl_width_to_depth(enc, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(enc); best_val = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
    width_fails = 0
    while width_fails < scfg.patience_width and can_widen(enc, scfg.ex_k, scfg):
        pre = snapshot(enc); enc.widen_all(scfg.ex_k)
        v = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(enc)
            depth_fails = 0
            while depth_fails < scfg.patience_depth and can_deepen(enc, scfg):
                pre2 = snapshot(enc); enc.append_depth()
                v2 = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(enc)
                else: depth_fails += 1; restore(enc, pre2)
        else:
            width_fails += 1; restore(enc, pre)
    restore(enc, best_snap); return enc

def cnn_ssl_depth_to_width(enc, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(enc); best_val = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
    depth_fails = 0
    while depth_fails < scfg.patience_depth and can_deepen(enc, scfg):
        pre = snapshot(enc); enc.append_depth()
        v = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(enc)
            width_fails = 0
            while width_fails < scfg.patience_width and can_widen(enc, scfg.ex_k, scfg):
                pre2 = snapshot(enc); enc.widen_all(scfg.ex_k)
                v2 = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(enc)
                else: width_fails += 1; restore(enc, pre2)
        else:
            depth_fails += 1; restore(enc, pre)
    restore(enc, best_snap); return enc

def cnn_ssl_alt_depth_first(enc, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(enc); best_val = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(enc, scfg) and ok(total):
            pre = snapshot(enc); enc.append_depth()
            v = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(enc); improved = True
            else: depth_fails += 1; restore(enc, pre)
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(enc, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(enc); enc.widen_all(scfg.ex_k)
            v = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(enc); improved = True
            else: width_fails += 1; restore(enc, pre)
    restore(enc, best_snap); return enc

def cnn_ssl_alt_width_first(enc, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(enc); best_val = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(enc, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(enc); enc.widen_all(scfg.ex_k)
            v = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(enc); improved = True
            else: width_fails += 1; restore(enc, pre)
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(enc, scfg) and ok(total):
            pre = snapshot(enc); enc.append_depth()
            v = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(enc); improved = True
            else: depth_fails += 1; restore(enc, pre)
    restore(enc, best_snap); return enc

def cnn_ssl_depth_only(enc, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(enc); best_val = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_depth and can_deepen(enc, scfg):
        pre = snapshot(enc); enc.append_depth()
        v = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(enc)
        else: fails += 1; restore(enc, pre)
    restore(enc, best_snap); return enc

def cnn_ssl_width_only(enc, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(enc); best_val = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_width and can_widen(enc, scfg.ex_k, scfg):
        pre = snapshot(enc); enc.widen_all(scfg.ex_k)
        v = _train_eval_val(enc, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(enc)
        else: fails += 1; restore(enc, pre)
    restore(enc, best_snap); return enc
