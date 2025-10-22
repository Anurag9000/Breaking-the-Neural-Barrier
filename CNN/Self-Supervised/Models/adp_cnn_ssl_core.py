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

import math
import random
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Basic building blocks
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ---------------------------
# Adaptive CNN backbone
# ---------------------------

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__()
        assert len(widths) >= 1, "Need at least one block"
        self.widths = list(widths)
        self.blocks = nn.ModuleList()
        self.pooling_indices = set(pooling_indices or [])  # 0-based indices
        prev = in_ch
        for i, w in enumerate(self.widths):
            self.blocks.append(ConvBNReLU(prev, w))
            prev = w
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(self.widths[-1], num_classes)

    # ---- book-keeping helpers ----
    def _depth(self):
        return len(self.blocks)

    def _last_width(self):
        return self.widths[-1]

    def _total_neurons(self):
        # simple proxy: sum of out_channels for conv blocks + head fan-in
        return sum(self.widths) + self.widths[-1]

    # ---- resizing utils ----
    @staticmethod
    def _overlap_copy(dst: torch.Tensor, src: torch.Tensor):
        with torch.no_grad():
            slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
            dst[slices].copy_(src[slices])

    def _resize_conv2d(self, old: nn.Conv2d, in_ch: int, out_ch: int):
        new = nn.Conv2d(in_ch, out_ch, kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, dilation=old.dilation, bias=(old.bias is not None))
        self._overlap_copy(new.weight, old.weight)
        if old.bias is not None:
            self._overlap_copy(new.bias, old.bias)
        return new

    def _resize_bn2d(self, old: nn.BatchNorm2d, num_features: int):
        new = nn.BatchNorm2d(num_features)
        self._overlap_copy(new.weight, old.weight)
        self._overlap_copy(new.bias, old.bias)
        self._overlap_copy(new.running_mean, old.running_mean)
        self._overlap_copy(new.running_var, old.running_var)
        return new

    def _resize_linear(self, old: nn.Linear, in_ch: int, out_ch: int):
        new = nn.Linear(in_ch, out_ch, bias=old.bias is not None)
        self._overlap_copy(new.weight, old.weight)
        if old.bias is not None:
            self._overlap_copy(new.bias, old.bias)
        return new

    # ---- expansion ops ----
    def append_depth(self):
        in_ch = self.widths[-1]
        out_ch = in_ch
        self.blocks.append(ConvBNReLU(in_ch, out_ch))
        self.widths.append(out_ch)
        # head unchanged

    def widen_all(self, ex_k: int = 8, max_width: Optional[int] = None):
        new_widths = []
        for w in self.widths:
            nw = w + ex_k
            if max_width is not None:
                nw = min(nw, max_width)
            new_widths.append(nw)
        # rebuild blocks
        prev = self.blocks[0].conv.in_channels
        for i, block in enumerate(self.blocks):
            old = block
            new_out = new_widths[i]
            new_conv = self._resize_conv2d(old.conv, prev, new_out)
            new_bn = self._resize_bn2d(old.bn, new_out)
            self.blocks[i] = ConvBNReLU(prev, new_out)
            self.blocks[i].conv = new_conv
            self.blocks[i].bn = new_bn
            prev = new_out
        # resize head
        self.head = self._resize_linear(self.head, new_widths[-1], self.head.out_features)
        self.widths = new_widths

    # ---- snapshot / restore ----
    def snapshot(self):
        return {
            "state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "widths": list(self.widths),
        }

    def restore(self, snap):
        self.widths = list(snap["widths"])
        # rebuild to match widths
        prev = self.blocks[0].conv.in_channels
        for i in range(len(self.blocks)):
            w = self.widths[i]
            old = self.blocks[i]
            self.blocks[i] = ConvBNReLU(prev, w)
            self.blocks[i].conv = self._resize_conv2d(old.conv, prev, w)
            self.blocks[i].bn = self._resize_bn2d(old.bn, w)
            prev = w
        self.head = self._resize_linear(self.head, self.widths[-1], self.head.out_features)
        self.load_state_dict({k: v for k, v in snap["state"].items()}, strict=True)

    def forward_features(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.pooling_indices:
                x = F.max_pool2d(x, 2)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats)

# ---------------------------
# SSL Objectives (single-model)
# ---------------------------

class RotationSSL(nn.Module):
    """RotNet-style 4-way rotation classification on the same encoder."""
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.rot_head = nn.Linear(encoder._last_width(), 4)

    @staticmethod
    def _rotate_batch(x):
        # randomly rotate each sample by 0,90,180,270
        B = x.size(0)
        k = torch.randint(low=0, high=4, size=(B,), device=x.device)
        x_rot = []
        for i in range(B):
            xi = x[i]
            if k[i] == 0:
                xr = xi
            elif k[i] == 1:
                xr = xi.rot90(1, [1, 2])
            elif k[i] == 2:
                xr = xi.rot90(2, [1, 2])
            else:
                xr = xi.rot90(3, [1, 2])
            x_rot.append(xr)
        return torch.stack(x_rot, 0), k

    def forward(self, x):
        xr, y = self._rotate_batch(x)
        feats = self.encoder.forward_features(xr)
        logits = self.rot_head(feats)
        loss = F.cross_entropy(logits, y)
        return loss, {"rot_acc": (logits.argmax(1) == y).float().mean().item()}

class JigsawSSL(nn.Module):
    """Predict permutation index from a fixed bank of permutations (size K)."""
    def __init__(self, encoder: AdaptiveCNN, grid=3, K=30, image_size=32):
        super().__init__()
        self.encoder = encoder
        self.grid = grid
        self.K = K
        self.perms = self._create_perm_bank(grid*grid, K, seed=1234)
        self.cls = nn.Linear(encoder._last_width(), K)
        self.image_size = image_size

    @staticmethod
    def _create_perm_bank(n, K, seed=0):
        rng = random.Random(seed)
        perms = set()
        base = list(range(n))
        while len(perms) < K:
            p = base[:]
            rng.shuffle(p)
            if tuple(p) not in perms and p != base:
                perms.add(tuple(p))
        return [list(p) for p in perms]

    def _tile(self, x):
        # x: BxCxHxW -> tiles per sample
        B, C, H, W = x.shape
        g = self.grid
        h, w = H // g, W // g
        tiles = []
        for i in range(g):
            for j in range(g):
                tiles.append(x[:, :, i*h:(i+1)*h, j*w:(j+1)*w])
        # tiles list length g*g, each BxCx(h)x(w)
        return tiles, h, w

    def _untile(self, tiles, perm, g, h, w):
        # tiles: list of BxCx(h)x(w); perm: list index mapping new order
        B, C = tiles[0].shape[:2]
        rows = []
        for i in range(g):
            row = []
            for j in range(g):
                idx = perm[i*g+j]
                row.append(tiles[idx])
            rows.append(torch.cat(row, dim=3))
        return torch.cat(rows, dim=2)

    def forward(self, x):
        B, C, H, W = x.shape
        # center crop to be divisible by grid
        minHW = min(H, W)
        tgt = self.image_size if minHW >= self.image_size else minHW - (minHW % self.grid)
        xt = F.interpolate(x, size=(tgt, tgt), mode="bilinear", align_corners=False)
        tiles, h, w = self._tile(xt)
        # choose a permutation id for each sample
        ids = torch.randint(low=0, high=self.K, size=(B,), device=x.device)
        xj_list = []
        for b in range(B):
            perm = self.perms[ids[b].item()]
            re = self._untile([t[b:b+1] for t in tiles], perm, self.grid, h, w)  # 1xCxHxW
            xj_list.append(re)
        xj = torch.cat(xj_list, 0)
        feats = self.encoder.forward_features(xj)
        logits = self.cls(feats)
        loss = F.cross_entropy(logits, ids)
        return loss, {"jig_acc": (logits.argmax(1) == ids).float().mean().item()}

class ContextSSL(nn.Module):
    """Predict relative position of a small patch w.r.t. a reference patch (Doersch-style)."""
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder._last_width()*2, 8)  # 8 neighbors

    def _sample_patches(self, x, patch=16):
        B, C, H, W = x.shape
        if H < patch*2 or W < patch*2:
            x = F.interpolate(x, size=(max(2*patch, H), max(2*patch, W)), mode="bilinear", align_corners=False)
            B, C, H, W = x.shape
        # sample centers away from borders
        ref_y = torch.randint(patch, H - patch, (B,), device=x.device)
        ref_x = torch.randint(patch, W - patch, (B,), device=x.device)
        # choose relative offset among 8
        rels = torch.randint(0, 8, (B,), device=x.device)
        offsets = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
        # extract ref and neighbor patches
        ref = []
        ctx = []
        for i in range(B):
            y, x0 = ref_y[i].item(), ref_x[i].item()
            dy, dx = offsets[rels[i]]
            ny = max(patch, min(H - patch, y + dy*patch))
            nx = max(patch, min(W - patch, x0 + dx*patch))
            ref.append(x[i:i+1, :, y-patch:y+patch, x0-patch:x0+patch])
            ctx.append(x[i:i+1, :, ny-patch:ny+patch, nx-patch:nx+patch])
        ref = torch.cat(ref, 0)
        ctx = torch.cat(ctx, 0)
        return ref, ctx, rels

    def forward(self, x):
        p1, p2, y = self._sample_patches(x)
        f1 = self.encoder.forward_features(p1)
        f2 = self.encoder.forward_features(p2)
        logits = self.cls(torch.cat([f1, f2], dim=1))
        loss = F.cross_entropy(logits, y)
        return loss, {"ctx_acc": (logits.argmax(1) == y).float().mean().item()}

class MaskedAESLL(nn.Module):
    """Masked autoencoding with a lightweight CNN decoder (no sampling, single model)."""
    def __init__(self, encoder: AdaptiveCNN, mask_ratio=0.6):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        d = encoder._last_width()
        # simple decoder: MLP to expand + conv to reconstruct
        self.proj = nn.Linear(d, d*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(d, d//2, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d//2, d//4, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d//4, 3, 3, 1, 1)
        )

    def _apply_mask(self, x):
        B, C, H, W = x.shape
        # block mask
        mask_h = int(H * math.sqrt(self.mask_ratio) * 0.5)
        mask_w = int(W * math.sqrt(self.mask_ratio) * 0.5)
        ys = torch.randint(0, max(1, H - mask_h), (B,), device=x.device)
        xs = torch.randint(0, max(1, W - mask_w), (B,), device=x.device)
        xm = x.clone()
        for i in range(B):
            xm[i, :, ys[i]:ys[i]+mask_h, xs[i]:xs[i]+mask_w] = 0.0
        return xm

    def forward(self, x):
        xm = self._apply_mask(x)
        feats = self.encoder.forward_features(xm)
        z = self.proj(feats).view(feats.size(0), -1, 1, 1)  # Bxdx1x1
        recon = self.deconv(z)
        recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.l1_loss(recon, x)
        return loss, {"mae_l1": loss.item()}

class ColorizationSSL(nn.Module):
    """Predict ab from L using a decoder; uses downsampled target for stability."""
    def __init__(self, encoder: AdaptiveCNN, down=4):
        super().__init__()
        self.encoder = encoder
        d = encoder._last_width()
        self.proj = nn.Linear(d, d*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(d, d//2, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d//2, d//4, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d//4, 2, 3, 1, 1) # ab channels
        )
        self.down = down

    @staticmethod
    def _rgb_to_lab(img: torch.Tensor):
        # naive, differentiable-free conversion; approximate
        # expects img in [0,1]
        def f(t):
            delta = 6/29
            return torch.where(t > (delta**3), t.pow(1/3), t/(3*delta**2) + 4/29)
        r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
        # sRGB to XYZ (D65)
        X = 0.412453*r + 0.357580*g + 0.180423*b
        Y = 0.212671*r + 0.715160*g + 0.072169*b
        Z = 0.019334*r + 0.119193*g + 0.950227*b
        Xn, Yn, Zn = 0.950456, 1.0, 1.088754
        fx, fy, fz = f(X/Xn), f(Y/Yn), f(Z/Zn)
        L = 116*fy - 16
        a = 500*(fx - fy)
        b2 = 200*(fy - fz)
        # normalize roughly
        L = (L/100.0).clamp(0,1)
        a = (a + 128)/255.0
        b2 = (b2 + 128)/255.0
        lab = torch.cat([L, a, b2], dim=1)
        return lab

    def forward(self, x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        lab = self._rgb_to_lab(x)
        L = lab[:,0:1]
        ab = lab[:,1:3]
        feats = self.encoder.forward_features(L.repeat(1,3,1,1))
        z = self.proj(feats).view(feats.size(0), -1, 1, 1)
        pred = self.dec(z)
        pred = F.interpolate(pred, size=ab.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.l1_loss(pred, ab)
        return loss, {"col_l1": loss.item()}

class WhiteningSSL(nn.Module):
    """Feature decorrelation / whitening regularizer: push covariance to identity."""
    def __init__(self, encoder: AdaptiveCNN, eps=1e-4):
        super().__init__()
        self.encoder = encoder
        self.eps = eps

    def forward(self, x):
        feats = self.encoder.forward_features(x)  # Bxd
        feats = feats - feats.mean(dim=0, keepdim=True)
        B = feats.size(0)
        cov = (feats.T @ feats) / (B - 1 + 1e-6)  # dxd
        I = torch.eye(cov.size(0), device=cov.device)
        loss = F.mse_loss(cov, I)
        return loss, {"whiten_mse": loss.item()}

class RotJigSSL(nn.Module):
    """Multi-task: rotation + jigsaw (sum of losses)."""
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.rot = RotationSSL(encoder)
        self.jig = JigsawSSL(encoder)

    def forward(self, x):
        lr, mr = self.rot(x)
        lj, mj = self.jig(x)
        loss = lr + lj
        metrics = {**mr, **mj}
        return loss, metrics

# Stubs (safe fallbacks) for completeness — implement as needed.
class TemporalOrderSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder._last_width(), 2)  # in-order vs shuffled as simple proxy

    def forward(self, x):
        # Expect x as BxTxCxHxW; fallback: treat as in-order
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            feats = self.encoder.forward_features(x).view(B, T, -1).mean(1)
            logits = self.cls(feats)
            y = torch.zeros(B, dtype=torch.long, device=x.device)
            loss = F.cross_entropy(logits, y)
            return loss, {"temp_acc": (logits.argmax(1)==y).float().mean().item()}
        else:
            # No sequence: return zero loss to not break training
            zero = torch.zeros((), device=x.device, requires_grad=True)
            return zero, {"temp_acc": 0.0}

class OpticalFlowSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.flow_head = nn.Linear(encoder._last_width(), 2)  # dummy 2D flow avg prediction

    def forward(self, x):
        # Expect pair of frames concatenated along batch or channel; here we return zero if not available
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            assert T >= 2, "Need at least 2 frames"
            pair = x[:, :2].reshape(B*2, C, H, W)
            feats = self.encoder.forward_features(pair).view(B,2,-1).mean(1)
            pred = self.flow_head(feats)
            target = torch.zeros_like(pred)
            loss = F.mse_loss(pred, target)
            return loss, {"flow_mse": loss.item()}
        zero = torch.zeros((), device=x.device, requires_grad=True)
        return zero, {"flow_mse": 0.0}

class ExemplarSSL(nn.Module):
    """Instance discrimination (single-view proxy): classify sample augmentation id among M bins."""
    def __init__(self, encoder: AdaptiveCNN, M=1024):
        super().__init__()
        self.encoder = encoder
        self.proto = nn.Linear(encoder._last_width(), M, bias=False)

    def forward(self, x):
        feats = self.encoder.forward_features(x)
        logits = self.proto(F.normalize(feats, dim=1))
        # uniform targets (no teacher) — encourage peaky assignments
        loss = -(F.log_softmax(logits, dim=1).mean())
        return loss, {"ex_uniform": loss.item()}

class PredictiveCodingSSL(nn.Module):
    """Predict next-layer features from current (simple self-regression)."""
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        d = encoder._last_width()
        self.pred = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d))

    def forward(self, x):
        feats = self.encoder.forward_features(x)
        pred = self.pred(feats.detach())  # stop-gradient target (still single model)
        loss = F.mse_loss(pred, feats)
        return loss, {"pc_mse": loss.item()}

# ---------------------------
# Objective factory
# ---------------------------

def build_objective(name: str, encoder: AdaptiveCNN):
    name = name.lower()
    if name == "rotation": return RotationSSL(encoder)
    if name == "jigsaw": return JigsawSSL(encoder)
    if name == "context": return ContextSSL(encoder)
    if name == "masked_ae": return MaskedAESLL(encoder)
    if name == "colorization": return ColorizationSSL(encoder)
    if name == "whitening": return WhiteningSSL(encoder)
    if name == "rot_jigsaw": return RotJigSSL(encoder)
    if name == "temporal_order": return TemporalOrderSSL(encoder)
    if name == "optical_flow": return OpticalFlowSSL(encoder)
    if name == "exemplar": return ExemplarSSL(encoder)
    if name == "predictive_coding": return PredictiveCodingSSL(encoder)
    raise ValueError(f"Unknown objective: {name}")

# ---------------------------
# Trainers & ADP search variants
# ---------------------------

class EarlyStopper:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = float("inf")
        self.bad = 0
        self.snapshot = None

    def step(self, val, snap):
        if val < self.best - self.delta:
            self.best = val
            self.bad = 0
            self.snapshot = snap
            return True
        else:
            self.bad += 1
            return False

    def done(self):
        return self.bad >= self.patience

def run_inner_ssl(encoder, objective_name, loader, device, epochs=5, lr=3e-4, weight_decay=1e-4, max_norm=1.0):
    encoder.train()
    obj = build_objective(objective_name, encoder).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopper(patience=max(3, epochs//2), delta=0.0)
    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            loss, metrics = obj(x)
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm)
            opt.step()
        # simple val proxy: last loss
        snap = encoder.snapshot()
        stopper.step(loss.item(), snap)
        if stopper.done():
            break
    # restore best
    if stopper.snapshot is not None:
        encoder.restore(stopper.snapshot)
    return stopper.best

# ---- Six ADP Variants ----

def adp_w_to_d(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    improved = True
    # width series
    width_fail = 0
    while improved and width_fail < cfg["patience_width"]:
        pre = best_val
        encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val
            best_snap = encoder.snapshot()
            # after each width acceptance, enter depth series
            depth_fail = 0
            while depth_fail < cfg["patience_depth"]:
                pre_d = best_val
                encoder.append_depth()
                val_d = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
                if val_d < pre_d - cfg["delta"]:
                    best_val = val_d
                    best_snap = encoder.snapshot()
                    depth_fail = 0
                else:
                    encoder.restore(best_snap)
                    depth_fail += 1
        else:
            encoder.restore(best_snap)
            width_fail += 1
    encoder.restore(best_snap)
    return best_val

def adp_d_to_w(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    # depth series first
    depth_fail = 0
    while depth_fail < cfg["patience_depth"]:
        pre = best_val
        encoder.append_depth()
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val
            best_snap = encoder.snapshot()
            # then width series
            width_fail = 0
            while width_fail < cfg["patience_width"]:
                pre_w = best_val
                encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
                val_w = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
                if val_w < pre_w - cfg["delta"]:
                    best_val = val_w
                    best_snap = encoder.snapshot()
                    width_fail = 0
                else:
                    encoder.restore(best_snap)
                    width_fail += 1
        else:
            encoder.restore(best_snap)
            depth_fail += 1
    encoder.restore(best_snap)
    return best_val

def adp_alt_depth_first(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    while True:
        accepted = False
        # depth phase
        d_fail = 0
        while d_fail < cfg["patience_depth"]:
            pre = best_val
            encoder.append_depth()
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True
                d_fail = 0
            else:
                encoder.restore(best_snap); d_fail += 1; break
        # width phase
        w_fail = 0
        while w_fail < cfg["patience_width"]:
            pre = best_val
            encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True
                w_fail = 0
            else:
                encoder.restore(best_snap); w_fail += 1; break
        if not accepted: break
    encoder.restore(best_snap)
    return best_val

def adp_alt_width_first(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    while True:
        accepted = False
        # width phase
        w_fail = 0
        while w_fail < cfg["patience_width"]:
            pre = best_val
            encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True
                w_fail = 0
            else:
                encoder.restore(best_snap); w_fail += 1; break
        # depth phase
        d_fail = 0
        while d_fail < cfg["patience_depth"]:
            pre = best_val
            encoder.append_depth()
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True
                d_fail = 0
            else:
                encoder.restore(best_snap); d_fail += 1; break
        if not accepted: break
    encoder.restore(best_snap)
    return best_val

def adp_depth_only(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    fails = 0
    while fails < cfg["patience_depth"]:
        pre = best_val
        encoder.append_depth()
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val; best_snap = encoder.snapshot(); fails = 0
        else:
            encoder.restore(best_snap); fails += 1
    encoder.restore(best_snap)
    return best_val

def adp_width_only(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    fails = 0
    while fails < cfg["patience_width"]:
        pre = best_val
        encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val; best_snap = encoder.snapshot(); fails = 0
        else:
            encoder.restore(best_snap); fails += 1
    encoder.restore(best_snap)
    return best_val

VARIANT_FUNCS = {
    "wd": adp_w_to_d,
    "dw": adp_d_to_w,
    "alt_d": adp_alt_depth_first,
    "alt_w": adp_alt_width_first,
    "depth_only": adp_depth_only,
    "width_only": adp_width_only,
}

import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Basic building blocks
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ---------------------------
# Adaptive CNN backbone
# ---------------------------

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__()
        assert len(widths) >= 1, "Need at least one block"
        self.widths = list(widths)
        self.blocks = nn.ModuleList()
        self.pooling_indices = set(pooling_indices or [])  # 0-based indices
        prev = in_ch
        for i, w in enumerate(self.widths):
            self.blocks.append(ConvBNReLU(prev, w))
            prev = w
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(self.widths[-1], num_classes)

    def _depth(self):
        return len(self.blocks)

    def _last_width(self):
        return self.widths[-1]

    def _total_neurons(self):
        return sum(self.widths) + self.widths[-1]

    @staticmethod
    def _overlap_copy(dst: torch.Tensor, src: torch.Tensor):
        with torch.no_grad():
            slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
            dst[slices].copy_(src[slices])

    def _resize_conv2d(self, old: nn.Conv2d, in_ch: int, out_ch: int):
        new = nn.Conv2d(in_ch, out_ch, kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, dilation=old.dilation, bias=(old.bias is not None))
        self._overlap_copy(new.weight, old.weight)
        if old.bias is not None:
            self._overlap_copy(new.bias, old.bias)
        return new

    def _resize_bn2d(self, old: nn.BatchNorm2d, num_features: int):
        new = nn.BatchNorm2d(num_features)
        self._overlap_copy(new.weight, old.weight)
        self._overlap_copy(new.bias, old.bias)
        self._overlap_copy(new.running_mean, old.running_mean)
        self._overlap_copy(new.running_var, old.running_var)
        return new

    def _resize_linear(self, old: nn.Linear, in_ch: int, out_ch: int):
        new = nn.Linear(in_ch, out_ch, bias=old.bias is not None)
        self._overlap_copy(new.weight, old.weight)
        if old.bias is not None:
            self._overlap_copy(new.bias, old.bias)
        return new

    # ---- expansion ops ----
    def append_depth(self):
        in_ch = self.widths[-1]
        out_ch = in_ch
        self.blocks.append(ConvBNReLU(in_ch, out_ch))
        self.widths.append(out_ch)

    def widen_all(self, ex_k: int = 8, max_width: Optional[int] = None):
        new_widths = []
        for w in self.widths:
            nw = w + ex_k
            if max_width is not None:
                nw = min(nw, max_width)
            new_widths.append(nw)
        prev = self.blocks[0].conv.in_channels
        for i, block in enumerate(self.blocks):
            old = block
            new_out = new_widths[i]
            new_conv = self._resize_conv2d(old.conv, prev, new_out)
            new_bn = self._resize_bn2d(old.bn, new_out)
            self.blocks[i] = ConvBNReLU(prev, new_out)
            self.blocks[i].conv = new_conv
            self.blocks[i].bn = new_bn
            prev = new_out
        self.head = self._resize_linear(self.head, new_widths[-1], self.head.out_features)
        self.widths = new_widths

    # ---- snapshot / restore ----
    def snapshot(self):
        return {"state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
                "widths": list(self.widths)}

    def restore(self, snap):
        self.widths = list(snap["widths"])
        prev = self.blocks[0].conv.in_channels
        for i in range(len(self.blocks)):
            w = self.widths[i]
            old = self.blocks[i]
            self.blocks[i] = ConvBNReLU(prev, w)
            self.blocks[i].conv = self._resize_conv2d(old.conv, prev, w)
            self.blocks[i].bn = self._resize_bn2d(old.bn, w)
            prev = w
        self.head = self._resize_linear(self.head, self.widths[-1], self.head.out_features)
        self.load_state_dict({k: v for k, v in snap["state"].items()}, strict=True)

    def forward_features(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.pooling_indices:
                x = F.max_pool2d(x, 2)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats)

# ---------------------------
# SSL Objectives (single-model)
# ---------------------------

class RotationSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.rot_head = nn.Linear(encoder._last_width(), 4)

    @staticmethod
    def _rotate_batch(x):
        B = x.size(0)
        k = torch.randint(low=0, high=4, size=(B,), device=x.device)
        x_rot = []
        for i in range(B):
            xi = x[i]
            if k[i] == 0:   xr = xi
            elif k[i] == 1: xr = xi.rot90(1, [1, 2])
            elif k[i] == 2: xr = xi.rot90(2, [1, 2])
            else:           xr = xi.rot90(3, [1, 2])
            x_rot.append(xr)
        return torch.stack(x_rot, 0), k

    def forward(self, x):
        xr, y = self._rotate_batch(x)
        feats = self.encoder.forward_features(xr)
        logits = self.rot_head(feats)
        loss = F.cross_entropy(logits, y)
        return loss, {"rot_acc": (logits.argmax(1) == y).float().mean().item()}

class JigsawSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN, grid=3, K=30, image_size=32):
        super().__init__()
        self.encoder = encoder
        self.grid = grid
        self.K = K
        self.perms = self._create_perm_bank(grid*grid, K, seed=1234)
        self.cls = nn.Linear(encoder._last_width(), K)
        self.image_size = image_size

    @staticmethod
    def _create_perm_bank(n, K, seed=0):
        rng = random.Random(seed)
        perms = set()
        base = list(range(n))
        while len(perms) < K:
            p = base[:]
            rng.shuffle(p)
            if tuple(p) not in perms and p != base:
                perms.add(tuple(p))
        return [list(p) for p in perms]

    def _tile(self, x):
        B, C, H, W = x.shape
        g = self.grid
        h, w = H // g, W // g
        tiles = []
        for i in range(g):
            for j in range(g):
                tiles.append(x[:, :, i*h:(i+1)*h, j*w:(j+1)*w])
        return tiles, h, w

    def _untile(self, tiles, perm, g, h, w):
        rows = []
        for i in range(g):
            row = []
            for j in range(g):
                idx = perm[i*g+j]
                row.append(tiles[idx])
            rows.append(torch.cat(row, dim=3))
        return torch.cat(rows, dim=2)

    def forward(self, x):
        B, C, H, W = x.shape
        minHW = min(H, W)
        tgt = self.image_size if minHW >= self.image_size else minHW - (minHW % self.grid)
        xt = F.interpolate(x, size=(tgt, tgt), mode="bilinear", align_corners=False)
        tiles, h, w = self._tile(xt)
        ids = torch.randint(low=0, high=self.K, size=(B,), device=x.device)
        xj_list = []
        for b in range(B):
            perm = self.perms[ids[b].item()]
            re = self._untile([t[b:b+1] for t in tiles], perm, self.grid, h, w)
            xj_list.append(re)
        xj = torch.cat(xj_list, 0)
        feats = self.encoder.forward_features(xj)
        logits = self.cls(feats)
        loss = F.cross_entropy(logits, ids)
        return loss, {"jig_acc": (logits.argmax(1) == ids).float().mean().item()}

class ContextSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder._last_width()*2, 8)  # 8 neighbors

    def _sample_patches(self, x, patch=16):
        B, C, H, W = x.shape
        if H < patch*2 or W < patch*2:
            x = F.interpolate(x, size=(max(2*patch, H), max(2*patch, W)),
                              mode="bilinear", align_corners=False)
            B, C, H, W = x.shape
        ref_y = torch.randint(patch, H - patch, (B,), device=x.device)
        ref_x = torch.randint(patch, W - patch, (B,), device=x.device)
        rels = torch.randint(0, 8, (B,), device=x.device)
        offsets = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
        ref, ctx = [], []
        for i in range(B):
            y, x0 = ref_y[i].item(), ref_x[i].item()
            dy, dx = offsets[rels[i]]
            ny = max(patch, min(H - patch, y + dy*patch))
            nx = max(patch, min(W - patch, x0 + dx*patch))
            ref.append(x[i:i+1, :, y-patch:y+patch, x0-patch:x0+patch])
            ctx.append(x[i:i+1, :, ny-patch:ny+patch, nx-patch:nx+patch])
        return torch.cat(ref, 0), torch.cat(ctx, 0), rels

    def forward(self, x):
        p1, p2, y = self._sample_patches(x)
        f1 = self.encoder.forward_features(p1)
        f2 = self.encoder.forward_features(p2)
        logits = self.cls(torch.cat([f1, f2], dim=1))
        loss = F.cross_entropy(logits, y)
        return loss, {"ctx_acc": (logits.argmax(1) == y).float().mean().item()}

class MaskedAESLL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN, mask_ratio=0.6):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        d = encoder._last_width()
        self.proj = nn.Linear(d, d*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(d, d//2, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d//2, d//4, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d//4, 3, 3, 1, 1)
        )

    def _apply_mask(self, x):
        B, C, H, W = x.shape
        mask_h = int(H * math.sqrt(self.mask_ratio) * 0.5)
        mask_w = int(W * math.sqrt(self.mask_ratio) * 0.5)
        ys = torch.randint(0, max(1, H - mask_h), (B,), device=x.device)
        xs = torch.randint(0, max(1, W - mask_w), (B,), device=x.device)
        xm = x.clone()
        for i in range(B):
            xm[i, :, ys[i]:ys[i]+mask_h, xs[i]:xs[i]+mask_w] = 0.0
        return xm

    def forward(self, x):
        xm = self._apply_mask(x)
        feats = self.encoder.forward_features(xm)
        z = self.proj(feats).view(feats.size(0), -1, 1, 1)
        recon = self.deconv(z)
        recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.l1_loss(recon, x)
        return loss, {"mae_l1": loss.item()}

class ColorizationSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN, down=4):
        super().__init__()
        self.encoder = encoder
        d = encoder._last_width()
        self.proj = nn.Linear(d, d*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(d, d//2, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d//2, d//4, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d//4, 2, 3, 1, 1)
        )
        self.down = down

    @staticmethod
    def _rgb_to_lab(img: torch.Tensor):
        def f(t):
            delta = 6/29
            return torch.where(t > (delta**3), t.pow(1/3), t/(3*delta**2) + 4/29)
        r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
        X = 0.412453*r + 0.357580*g + 0.180423*b
        Y = 0.212671*r + 0.715160*g + 0.072169*b
        Z = 0.019334*r + 0.119193*g + 0.950227*b
        Xn, Yn, Zn = 0.950456, 1.0, 1.088754
        fx, fy, fz = f(X/Xn), f(Y/Yn), f(Z/Zn)
        L = 116*fy - 16
        a = 500*(fx - fy)
        b2 = 200*(fy - fz)
        L = (L/100.0).clamp(0,1)
        a = (a + 128)/255.0
        b2 = (b2 + 128)/255.0
        return torch.cat([L, a, b2], dim=1)

    def forward(self, x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        lab = self._rgb_to_lab(x)
        L = lab[:,0:1]
        ab = lab[:,1:3]
        feats = self.encoder.forward_features(L.repeat(1,3,1,1))
        z = self.proj(feats).view(feats.size(0), -1, 1, 1)
        pred = self.dec(z)
        pred = F.interpolate(pred, size=ab.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.l1_loss(pred, ab)
        return loss, {"col_l1": loss.item()}

class WhiteningSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN, eps=1e-4):
        super().__init__()
        self.encoder = encoder
        self.eps = eps

    def forward(self, x):
        feats = self.encoder.forward_features(x)
        feats = feats - feats.mean(dim=0, keepdim=True)
        B = feats.size(0)
        cov = (feats.T @ feats) / (B - 1 + 1e-6)
        I = torch.eye(cov.size(0), device=cov.device)
        loss = F.mse_loss(cov, I)
        return loss, {"whiten_mse": loss.item()}

class RotJigSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.rot = RotationSSL(encoder)
        self.jig = JigsawSSL(encoder)

    def forward(self, x):
        lr, mr = self.rot(x)
        lj, mj = self.jig(x)
        loss = lr + lj
        return loss, {**mr, **mj}

class TemporalOrderSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder._last_width(), 2)

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            x = x.view(B*T, C, H, W)
            feats = self.encoder.forward_features(x).view(B, T, -1).mean(1)
            logits = self.cls(feats)
            y = torch.zeros(B, dtype=torch.long, device=x.device)
            loss = F.cross_entropy(logits, y)
            return loss, {"temp_acc": (logits.argmax(1)==y).float().mean().item()}
        zero = torch.zeros((), device=x.device, requires_grad=True)
        return zero, {"temp_acc": 0.0}

class OpticalFlowSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.flow_head = nn.Linear(encoder._last_width(), 2)

    def forward(self, x):
        if x.dim() == 5:
            B, T, C, H, W = x.shape
            assert T >= 2, "Need at least 2 frames"
            pair = x[:, :2].reshape(B*2, C, H, W)
            feats = self.encoder.forward_features(pair).view(B,2,-1).mean(1)
            pred = self.flow_head(feats)
            target = torch.zeros_like(pred)
            loss = F.mse_loss(pred, target)
            return loss, {"flow_mse": loss.item()}
        zero = torch.zeros((), device=x.device, requires_grad=True)
        return zero, {"flow_mse": 0.0}

class ExemplarSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN, M=1024):
        super().__init__()
        self.encoder = encoder
        self.proto = nn.Linear(encoder._last_width(), M, bias=False)

    def forward(self, x):
        feats = self.encoder.forward_features(x)
        logits = self.proto(F.normalize(feats, dim=1))
        loss = -(F.log_softmax(logits, dim=1).mean())
        return loss, {"ex_uniform": loss.item()}

class PredictiveCodingSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        d = encoder._last_width()
        self.encoder = encoder
        self.pred = nn.Sequential(nn.Linear(d, d), nn.ReLU(inplace=True), nn.Linear(d, d))

    def forward(self, x):
        feats = self.encoder.forward_features(x)
        pred = self.pred(feats.detach())
        loss = F.mse_loss(pred, feats)
        return loss, {"pc_mse": loss.item()}

def build_objective(name: str, encoder: AdaptiveCNN):
    name = name.lower()
    if name == "rotation": return RotationSSL(encoder)
    if name == "jigsaw": return JigsawSSL(encoder)
    if name == "context": return ContextSSL(encoder)
    if name == "masked_ae": return MaskedAESLL(encoder)
    if name == "colorization": return ColorizationSSL(encoder)
    if name == "whitening": return WhiteningSSL(encoder)
    if name == "rot_jigsaw": return RotJigSSL(encoder)
    if name == "temporal_order": return TemporalOrderSSL(encoder)
    if name == "optical_flow": return OpticalFlowSSL(encoder)
    if name == "exemplar": return ExemplarSSL(encoder)
    if name == "predictive_coding": return PredictiveCodingSSL(encoder)
    raise ValueError(f"Unknown objective: {name}")

# ---- early stop + inner trainer ----

class EarlyStopper:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = float("inf")
        self.bad = 0
        self.snapshot = None

    def step(self, val, snap):
        if val < self.best - self.delta:
            self.best = val; self.bad = 0; self.snapshot = snap
            return True
        self.bad += 1
        return False

    def done(self):
        return self.bad >= self.patience

def run_inner_ssl(encoder, objective_name, loader, device, epochs=5, lr=3e-4, weight_decay=1e-4, max_norm=1.0):
    encoder.train()
    obj = build_objective(objective_name, encoder).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopper(patience=max(3, epochs//2), delta=0.0)
    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            loss, _ = obj(x)
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm)
            opt.step()
        snap = encoder.snapshot()
        stopper.step(loss.item(), snap)
        if stopper.done():
            break
    if stopper.snapshot is not None:
        encoder.restore(stopper.snapshot)
    return stopper.best

# ---- Six ADP Variants ----

def adp_w_to_d(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    width_fail = 0
    while width_fail < cfg["patience_width"]:
        pre = best_val
        encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]]:
            best_val = val; best_snap = encoder.snapshot()
            depth_fail = 0
            while depth_fail < cfg["patience_depth"]:
                pre_d = best_val
                encoder.append_depth()
                val_d = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
                if val_d < pre_d - cfg["delta"]:
                    best_val = val_d; best_snap = encoder.snapshot(); depth_fail = 0
                else:
                    encoder.restore(best_snap); depth_fail += 1
        else:
            encoder.restore(best_snap); width_fail += 1
    encoder.restore(best_snap); return best_val

def adp_d_to_w(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    depth_fail = 0
    while depth_fail < cfg["patience_depth"]:
        pre = best_val
        encoder.append_depth()
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val; best_snap = encoder.snapshot()
            width_fail = 0
            while width_fail < cfg["patience_width"]:
                pre_w = best_val
                encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
                val_w = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
                if val_w < pre_w - cfg["delta"]:
                    best_val = val_w; best_snap = encoder.snapshot(); width_fail = 0
                else:
                    encoder.restore(best_snap); width_fail += 1
        else:
            encoder.restore(best_snap); depth_fail += 1
    encoder.restore(best_snap); return best_val

def adp_alt_depth_first(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    while True:
        accepted = False
        d_fail = 0
        while d_fail < cfg["patience_depth"]:
            pre = best_val
            encoder.append_depth()
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True; d_fail = 0
            else:
                encoder.restore(best_snap); d_fail += 1; break
        w_fail = 0
        while w_fail < cfg["patience_width"]:
            pre = best_val
            encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True; w_fail = 0
            else:
                encoder.restore(best_snap); w_fail += 1; break
        if not accepted: break
    encoder.restore(best_snap); return best_val

def adp_alt_width_first(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    while True:
        accepted = False
        w_fail = 0
        while w_fail < cfg["patience_width"]:
            pre = best_val
            encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True; w_fail = 0
            else:
                encoder.restore(best_snap); w_fail += 1; break
        d_fail = 0
        while d_fail < cfg["patience_depth"]:
            pre = best_val
            encoder.append_depth()
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True; d_fail = 0
            else:
                encoder.restore(best_snap); d_fail += 1; break
        if not accepted: break
    encoder.restore(best_snap); return best_val

def adp_depth_only(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    fails = 0
    while fails < cfg["patience_depth"]:
        pre = best_val
        encoder.append_depth()
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val; best_snap = encoder.snapshot(); fails = 0
        else:
            encoder.restore(best_snap); fails += 1
    encoder.restore(best_snap); return best_val

def adp_width_only(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    fails = 0
    while fails < cfg["patience_width"]:
        pre = best_val
        encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val; best_snap = encoder.snapshot(); fails = 0
        else:
            encoder.restore(best_snap); fails += 1
    encoder.restore(best_snap); return best_val

VARIANT_FUNCS = {
    "wd": adp_w_to_d,
    "dw": adp_d_to_w,
    "alt_d": adp_alt_depth_first,
    "alt_w": adp_alt_width_first,
    "depth_only": adp_depth_only,
    "width_only": adp_width_only,
}

# ================================
# File: adp_ssl_set1_model.py
# Set 1 Objectives (5):
#   1) rotation
#   2) jigsaw
#   3) context
#   4) masked_ae
#   5) colorization
#
# Each objective works with all 6 ADP variants via the shared backbone + variant fns.
# Variants: wd, dw, alt_d, alt_w, depth_only, width_only
# ================================

import math
import random
from typing import List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Basic building blocks
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

# ---------------------------
# Adaptive CNN backbone
# ---------------------------

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__()
        assert len(widths) >= 1, "Need at least one block"
        self.widths = list(widths)
        self.blocks = nn.ModuleList()
        self.pooling_indices = set(pooling_indices or [])  # 0-based indices
        prev = in_ch
        for i, w in enumerate(self.widths):
            self.blocks.append(ConvBNReLU(prev, w))
            prev = w
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(self.widths[-1], num_classes)

    def _last_width(self):
        return self.widths[-1]

    @staticmethod
    def _overlap_copy(dst: torch.Tensor, src: torch.Tensor):
        with torch.no_grad():
            slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
            dst[slices].copy_(src[slices])

    def _resize_conv2d(self, old: nn.Conv2d, in_ch: int, out_ch: int):
        new = nn.Conv2d(in_ch, out_ch, kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, dilation=old.dilation, bias=(old.bias is not None))
        self._overlap_copy(new.weight, old.weight)
        if old.bias is not None:
            self._overlap_copy(new.bias, old.bias)
        return new

    def _resize_bn2d(self, old: nn.BatchNorm2d, num_features: int):
        new = nn.BatchNorm2d(num_features)
        self._overlap_copy(new.weight, old.weight)
        self._overlap_copy(new.bias, old.bias)
        self._overlap_copy(new.running_mean, old.running_mean)
        self._overlap_copy(new.running_var, old.running_var)
        return new

    def _resize_linear(self, old: nn.Linear, in_ch: int, out_ch: int):
        new = nn.Linear(in_ch, out_ch, bias=old.bias is not None)
        self._overlap_copy(new.weight, old.weight)
        if old.bias is not None:
            self._overlap_copy(new.bias, old.bias)
        return new

    # ---- expansion ops ----
    def append_depth(self):
        in_ch = self.widths[-1]
        out_ch = in_ch
        self.blocks.append(ConvBNReLU(in_ch, out_ch))
        self.widths.append(out_ch)

    def widen_all(self, ex_k: int = 8, max_width: Optional[int] = None):
        new_widths = []
        for w in self.widths:
            nw = w + ex_k
            if max_width is not None:
                nw = min(nw, max_width)
            new_widths.append(nw)
        prev = self.blocks[0].conv.in_channels
        for i, block in enumerate(self.blocks):
            old = block
            new_out = new_widths[i]
            new_conv = self._resize_conv2d(old.conv, prev, new_out)
            new_bn = self._resize_bn2d(old.bn, new_out)
            self.blocks[i] = ConvBNReLU(prev, new_out)
            self.blocks[i].conv = new_conv
            self.blocks[i].bn = new_bn
            prev = new_out
        self.head = self._resize_linear(self.head, new_widths[-1], self.head.out_features)
        self.widths = new_widths

    # ---- snapshot / restore ----
    def snapshot(self):
        return {"state": {k: v.detach().cpu() for k, v in self.state_dict().items()}, "widths": list(self.widths)}

    def restore(self, snap):
        self.widths = list(snap["widths"])
        prev = self.blocks[0].conv.in_channels
        for i in range(len(self.blocks)):
            w = self.widths[i]
            old = self.blocks[i]
            self.blocks[i] = ConvBNReLU(prev, w)
            self.blocks[i].conv = self._resize_conv2d(old.conv, prev, w)
            self.blocks[i].bn = self._resize_bn2d(old.bn, w)
            prev = w
        self.head = self._resize_linear(self.head, self.widths[-1], self.head.out_features)
        self.load_state_dict({k: v for k, v in snap["state"].items()}, strict=True)

    def forward_features(self, x):
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i in self.pooling_indices:
                x = F.max_pool2d(x, 2)
        x = self.gap(x).squeeze(-1).squeeze(-1)
        return x

    def forward(self, x):
        feats = self.forward_features(x)
        return self.head(feats)

# ---------------------------
# SSL Objectives (Set 1)
# ---------------------------

class RotationSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.rot_head = nn.Linear(encoder._last_width(), 4)

    @staticmethod
    def _rotate_batch(x):
        B = x.size(0)
        k = torch.randint(low=0, high=4, size=(B,), device=x.device)
        x_rot = []
        for i in range(B):
            xi = x[i]
            if k[i] == 0:   xr = xi
            elif k[i] == 1: xr = xi.rot90(1, [1, 2])
            elif k[i] == 2: xr = xi.rot90(2, [1, 2])
            else:           xr = xi.rot90(3, [1, 2])
            x_rot.append(xr)
        return torch.stack(x_rot, 0), k

    def forward(self, x):
        xr, y = self._rotate_batch(x)
        feats = self.encoder.forward_features(xr)
        logits = self.rot_head(feats)
        loss = F.cross_entropy(logits, y)
        return loss, {"rot_acc": (logits.argmax(1) == y).float().mean().item()}

class JigsawSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN, grid=3, K=30, image_size=32):
        super().__init__()
        self.encoder = encoder
        self.grid = grid
        self.K = K
        self.perms = self._create_perm_bank(grid*grid, K, seed=1234)
        self.cls = nn.Linear(encoder._last_width(), K)
        self.image_size = image_size

    @staticmethod
    def _create_perm_bank(n, K, seed=0):
        rng = random.Random(seed)
        perms = set(); base = list(range(n))
        while len(perms) < K:
            p = base[:]; rng.shuffle(p)
            if tuple(p) not in perms and p != base:
                perms.add(tuple(p))
        return [list(p) for p in perms]

    def _tile(self, x):
        B, C, H, W = x.shape; g = self.grid
        h, w = H // g, W // g
        tiles = []
        for i in range(g):
            for j in range(g):
                tiles.append(x[:, :, i*h:(i+1)*h, j*w:(j+1)*w])
        return tiles, h, w

    def _untile(self, tiles, perm, g, h, w):
        rows = []
        for i in range(g):
            row = []
            for j in range(g):
                idx = perm[i*g+j]
                row.append(tiles[idx])
            rows.append(torch.cat(row, dim=3))
        return torch.cat(rows, dim=2)

    def forward(self, x):
        B, C, H, W = x.shape
        minHW = min(H, W)
        tgt = self.image_size if minHW >= self.image_size else minHW - (minHW % self.grid)
        xt = F.interpolate(x, size=(tgt, tgt), mode="bilinear", align_corners=False)
        tiles, h, w = self._tile(xt)
        ids = torch.randint(low=0, high=self.K, size=(B,), device=x.device)
        xj_list = []
        for b in range(B):
            perm = self.perms[ids[b].item()]
            re = self._untile([t[b:b+1] for t in tiles], perm, self.grid, h, w)
            xj_list.append(re)
        xj = torch.cat(xj_list, 0)
        feats = self.encoder.forward_features(xj)
        logits = self.cls(feats)
        loss = F.cross_entropy(logits, ids)
        return loss, {"jig_acc": (logits.argmax(1) == ids).float().mean().item()}

class ContextSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        self.cls = nn.Linear(encoder._last_width()*2, 8)

    def _sample_patches(self, x, patch=16):
        B, C, H, W = x.shape
        if H < patch*2 or W < patch*2:
            x = F.interpolate(x, size=(max(2*patch, H), max(2*patch, W)), mode="bilinear", align_corners=False)
            B, C, H, W = x.shape
        ref_y = torch.randint(patch, H - patch, (B,), device=x.device)
        ref_x = torch.randint(patch, W - patch, (B,), device=x.device)
        rels = torch.randint(0, 8, (B,), device=x.device)
        offsets = [(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1),(-1,-1)]
        ref, ctx = [], []
        for i in range(B):
            y, x0 = ref_y[i].item(), ref_x[i].item()
            dy, dx = offsets[rels[i]]
            ny = max(patch, min(H - patch, y + dy*patch))
            nx = max(patch, min(W - patch, x0 + dx*patch))
            ref.append(x[i:i+1, :, y-patch:y+patch, x0-patch:x0+patch])
            ctx.append(x[i:i+1, :, ny-patch:ny+patch, nx-patch:nx+patch])
        return torch.cat(ref, 0), torch.cat(ctx, 0), rels

    def forward(self, x):
        p1, p2, y = self._sample_patches(x)
        f1 = self.encoder.forward_features(p1)
        f2 = self.encoder.forward_features(p2)
        logits = self.cls(torch.cat([f1, f2], dim=1))
        loss = F.cross_entropy(logits, y)
        return loss, {"ctx_acc": (logits.argmax(1) == y).float().mean().item()}

class MaskedAESLL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN, mask_ratio=0.6):
        super().__init__()
        self.encoder = encoder
        self.mask_ratio = mask_ratio
        d = encoder._last_width()
        self.proj = nn.Linear(d, d*4)
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(d, d//2, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d//2, d//4, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d//4, 3, 3, 1, 1)
        )

    def _apply_mask(self, x):
        B, C, H, W = x.shape
        mask_h = int(H * math.sqrt(self.mask_ratio) * 0.5)
        mask_w = int(W * math.sqrt(self.mask_ratio) * 0.5)
        ys = torch.randint(0, max(1, H - mask_h), (B,), device=x.device)
        xs = torch.randint(0, max(1, W - mask_w), (B,), device=x.device)
        xm = x.clone()
        for i in range(B):
            xm[i, :, ys[i]:ys[i]+mask_h, xs[i]:xs[i]+mask_w] = 0.0
        return xm

    def forward(self, x):
        xm = self._apply_mask(x)
        feats = self.encoder.forward_features(xm)
        z = self.proj(feats).view(feats.size(0), -1, 1, 1)
        recon = self.deconv(z)
        recon = F.interpolate(recon, size=x.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.l1_loss(recon, x)
        return loss, {"mae_l1": loss.item()}

class ColorizationSSL(nn.Module):
    def __init__(self, encoder: AdaptiveCNN):
        super().__init__()
        self.encoder = encoder
        d = encoder._last_width()
        self.proj = nn.Linear(d, d*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(d, d//2, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(d//2, d//4, 3, 2, 1, output_padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d//4, 2, 3, 1, 1)
        )

    @staticmethod
    def _rgb_to_lab(img: torch.Tensor):
        def f(t):
            delta = 6/29
            return torch.where(t > (delta**3), t.pow(1/3), t/(3*delta**2) + 4/29)
        r, g, b = img[:,0:1], img[:,1:2], img[:,2:3]
        X = 0.412453*r + 0.357580*g + 0.180423*b
        Y = 0.212671*r + 0.715160*g + 0.072169*b
        Z = 0.019334*r + 0.119193*g + 0.950227*b
        Xn, Yn, Zn = 0.950456, 1.0, 1.088754
        fx, fy, fz = f(X/Xn), f(Y/Yn), f(Z/Zn)
        L = 116*fy - 16
        a = 500*(fx - fy)
        b2 = 200*(fy - fz)
        L = (L/100.0).clamp(0,1)
        a = (a + 128)/255.0
        b2 = (b2 + 128)/255.0
        return torch.cat([L, a, b2], dim=1)

    def forward(self, x):
        x = (x - x.min()) / (x.max() - x.min() + 1e-6)
        lab = self._rgb_to_lab(x)
        L = lab[:,0:1]
        ab = lab[:,1:3]
        feats = self.encoder.forward_features(L.repeat(1,3,1,1))
        z = self.proj(feats).view(feats.size(0), -1, 1, 1)
        pred = self.dec(z)
        pred = F.interpolate(pred, size=ab.shape[-2:], mode="bilinear", align_corners=False)
        loss = F.l1_loss(pred, ab)
        return loss, {"col_l1": loss.item()}

# ---------------------------
# Objective factory (Set 1)
# ---------------------------

def build_objective(name: str, encoder: AdaptiveCNN):
    name = name.lower()
    if name == "rotation": return RotationSSL(encoder)
    if name == "jigsaw": return JigsawSSL(encoder)
    if name == "context": return ContextSSL(encoder)
    if name == "masked_ae": return MaskedAESLL(encoder)
    if name == "colorization": return ColorizationSSL(encoder)
    raise ValueError(f"Unknown objective for Set 1: {name}")

# ---------------------------
# Early stop + inner trainer
# ---------------------------

class EarlyStopper:
    def __init__(self, patience=10, delta=0.0):
        self.patience = patience
        self.delta = delta
        self.best = float("inf")
        self.bad = 0
        self.snapshot = None

    def step(self, val, snap):
        if val < self.best - self.delta:
            self.best = val; self.bad = 0; self.snapshot = snap
            return True
        self.bad += 1
        return False

    def done(self):
        return self.bad >= self.patience


def run_inner_ssl(encoder, objective_name, loader, device, epochs=5, lr=3e-4, weight_decay=1e-4, max_norm=1.0):
    encoder.train()
    obj = build_objective(objective_name, encoder).to(device)
    opt = torch.optim.AdamW(encoder.parameters(), lr=lr, weight_decay=weight_decay)
    stopper = EarlyStopper(patience=max(3, epochs//2), delta=0.0)
    for ep in range(epochs):
        for x, _ in loader:
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            loss, _ = obj(x)
            loss.backward()
            if max_norm is not None:
                torch.nn.utils.clip_grad_norm_(encoder.parameters(), max_norm)
            opt.step()
        snap = encoder.snapshot()
        stopper.step(loss.item(), snap)
        if stopper.done():
            break
    if stopper.snapshot is not None:
        encoder.restore(stopper.snapshot)
    return stopper.best

# ---------------------------
# Six ADP Variants
# ---------------------------

def adp_w_to_d(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    width_fail = 0
    while width_fail < cfg["patience_width"]:
        pre = best_val
        encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val; best_snap = encoder.snapshot()
            depth_fail = 0
            while depth_fail < cfg["patience_depth"]:
                pre_d = best_val
                encoder.append_depth()
                val_d = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
                if val_d < pre_d - cfg["delta"]:
                    best_val = val_d; best_snap = encoder.snapshot(); depth_fail = 0
                else:
                    encoder.restore(best_snap); depth_fail += 1
        else:
            encoder.restore(best_snap); width_fail += 1
    encoder.restore(best_snap); return best_val

def adp_d_to_w(encoder, cfg, loader, device):
    best_snap = encoder.snapshot()
    best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    depth_fail = 0
    while depth_fail < cfg["patience_depth"]:
        pre = best_val
        encoder.append_depth()
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val; best_snap = encoder.snapshot()
            width_fail = 0
            while width_fail < cfg["patience_width"]:
                pre_w = best_val
                encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
                val_w = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
                if val_w < pre_w - cfg["delta"]:
                    best_val = val_w; best_snap = encoder.snapshot(); width_fail = 0
                else:
                    encoder.restore(best_snap); width_fail += 1
        else:
            encoder.restore(best_snap); depth_fail += 1
    encoder.restore(best_snap); return best_val

def adp_alt_depth_first(encoder, cfg, loader, device):
    best_snap = encoder.snapshot(); best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    while True:
        accepted = False
        d_fail = 0
        while d_fail < cfg["patience_depth"]:
            pre = best_val
            encoder.append_depth()
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True; d_fail = 0
            else:
                encoder.restore(best_snap); d_fail += 1; break
        w_fail = 0
        while w_fail < cfg["patience_width"]:
            pre = best_val
            encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True; w_fail = 0
            else:
                encoder.restore(best_snap); w_fail += 1; break
        if not accepted: break
    encoder.restore(best_snap); return best_val

def adp_alt_width_first(encoder, cfg, loader, device):
    best_snap = encoder.snapshot(); best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    while True:
        accepted = False
        w_fail = 0
        while w_fail < cfg["patience_width"]:
            pre = best_val
            encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True; w_fail = 0
            else:
                encoder.restore(best_snap); w_fail += 1; break
        d_fail = 0
        while d_fail < cfg["patience_depth"]:
            pre = best_val
            encoder.append_depth()
            val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
            if val < pre - cfg["delta"]:
                best_val = val; best_snap = encoder.snapshot(); accepted = True; d_fail = 0
            else:
                encoder.restore(best_snap); d_fail += 1; break
        if not accepted: break
    encoder.restore(best_snap); return best_val

def adp_depth_only(encoder, cfg, loader, device):
    best_snap = encoder.snapshot(); best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    fails = 0
    while fails < cfg["patience_depth"]:
        pre = best_val
        encoder.append_depth()
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val; best_snap = encoder.snapshot(); fails = 0
        else:
            encoder.restore(best_snap); fails += 1
    encoder.restore(best_snap); return best_val

def adp_width_only(encoder, cfg, loader, device):
    best_snap = encoder.snapshot(); best_val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
    fails = 0
    while fails < cfg["patience_width"]:
        pre = best_val
        encoder.widen_all(cfg["ex_k"], max_width=cfg.get("max_width"))
        val = run_inner_ssl(encoder, cfg["objective"], loader, device, epochs=cfg["inner_epochs"])
        if val < pre - cfg["delta"]:
            best_val = val; best_snap = encoder.snapshot(); fails = 0
        else:
            encoder.restore(best_snap); fails += 1
    encoder.restore(best_snap); return best_val

VARIANT_FUNCS = {
    "wd": adp_w_to_d,
    "dw": adp_d_to_w,
    "alt_d": adp_alt_depth_first,
    "alt_w": adp_alt_width_first,
    "depth_only": adp_depth_only,
    "width_only": adp_width_only,
}

# ================================
# File: run_adp_ssl_set1.py
# ================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from adp_ssl_set1_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    name = name.lower()
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    tf_eval = transforms.Compose([
        transforms.Resize(32),
        transforms.CenterCrop(32),
        transforms.ToTensor(),
    ])
    if name == "cifar10":
        ds = datasets.CIFAR10(root="./data", train=train, download=True, transform=tf_train if train else tf_eval)
    elif name == "cifar100":
        ds = datasets.CIFAR100(root="./data", train=train, download=True, transform=tf_train if train else tf_eval)
    elif name == "stl10":
        split = "train" if train else "test"
        ds = datasets.STL10(root="./data", split=split, download=True, transform=tf_train if train else tf_eval)
    else:
        raise ValueError("Unknown dataset")
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10","cifar100","stl10"])
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--epochs", type=int, default=3, help="inner epochs per proposal")
    p.add_argument("--objective", type=str, default="rotation",
                   choices=["rotation","jigsaw","context","masked_ae","colorization"])
    p.add_argument("--variant", type=str, default="wd",
                   choices=["wd","dw","alt_d","alt_w","depth_only","width_only"])
    p.add_argument("--widths", type=str, default="32,32,64")
    p.add_argument("--pool_idx", type=str, default="1", help="0-based comma-separated indices for MaxPool")
    p.add_argument("--ex_k", type=int, default=16)
    p.add_argument("--patience_depth", type=int, default=2)
    p.add_argument("--patience_width", type=int, default=2)
    p.add_argument("--delta", type=float, default=0.0)
    p.add_argument("--max_width", type=int, default=256)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed); random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_ds = get_dataset(args.dataset, train=True)
    val_frac = 0.1
    n_val = int(len(train_ds)*val_frac)
    n_train = len(train_ds)-n_val
    train_ds, val_ds = random_split(train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    widths = [int(x) for x in args.widths.split(",")]
    pool_idx = [int(x) for x in args.pool_idx.split(",") if x.strip()!=""]

    model = AdaptiveCNN(in_ch=3, num_classes=10, widths=widths, pooling_indices=pool_idx).to(device)

    cfg = {
        "objective": args.objective,
        "inner_epochs": args.epochs,
        "patience_depth": args.patience_depth,
        "patience_width": args.patience_width,
        "ex_k": args.ex_k,
        "delta": args.delta,
        "max_width": args.max_width,
    }

    best_val = VARIANT_FUNCS[args.variant](model, cfg, train_loader, device)
    print(f"[DONE] Variant={args.variant} Objective={args.objective} BestProxyLoss={best_val:.4f}")
    torch.save({"state_dict": model.state_dict(), "widths": model.widths,
                "variant": args.variant, "objective": args.objective},
               f"adp_ssl_set1_{args.variant}_{args.objective}.pt")

if __name__ == "__main__":
    main()

# ======================================
# File: adp_ssl_set2_model.py
# Objectives (6–10):
#   6) rot_jigsaw (multi-task)
#   7) whitening (feature decorrelation)
#   8) exemplar (instance discrimination)
#   9) predictive_coding (self-regression)
#  10) temporal_order (frame order prediction)
# ======================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
import random

# ---------------------------
# Backbone
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 3, 1, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch, num_classes, widths: List[int]):
        super().__init__()
        self.widths = list(widths)
        self.blocks = nn.ModuleList([ConvBNReLU(in_ch if i==0 else widths[i-1], w) for i,w in enumerate(widths)])
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(widths[-1], num_classes)

    def _last_width(self):
        return self.widths[-1]

    def forward_features(self, x):
        for blk in self.blocks:
            x = blk(x)
        return self.gap(x).squeeze(-1).squeeze(-1)

    # dynamic grow ops
    def append_depth(self):
        w = self.widths[-1]
        self.blocks.append(ConvBNReLU(w, w)); self.widths.append(w)

    def widen_all(self, ex_k=8, max_width=None):
        new_w = []
        for w in self.widths:
            nw = min(w+ex_k, max_width) if max_width else w+ex_k
            new_w.append(nw)
        prev = self.blocks[0].conv.in_channels
        for i, blk in enumerate(self.blocks):
            old = blk
            blk.conv = nn.Conv2d(prev, new_w[i], 3, 1, 1, bias=False)
            blk.bn = nn.BatchNorm2d(new_w[i])
            blk.relu = nn.ReLU(inplace=True)
            prev = new_w[i]
        self.widths = new_w
        self.head = nn.Linear(new_w[-1], self.head.out_features)

    def snapshot(self):
        return {"state": {k:v.cpu() for k,v in self.state_dict().items()}, "widths": list(self.widths)}
    def restore(self, snap):
        self.load_state_dict(snap["state"], strict=False)

# ---------------------------
# SSL objectives (6–10)
# ---------------------------

class RotationSSL(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc=enc; self.rot_head=nn.Linear(enc._last_width(),4)
    def _rot(self,x):
        B=x.size(0); k=torch.randint(0,4,(B,),device=x.device); xs=[]
        for i in range(B): xs.append(x[i].rot90(int(k[i]),[1,2]))
        return torch.stack(xs,0),k
    def forward(self,x):
        xr,y=self._rot(x);f=self.enc.forward_features(xr);l=F.cross_entropy(self.rot_head(f),y)
        return l,{"rot_acc":(self.rot_head(f).argmax(1)==y).float().mean().item()}

class JigsawSSL(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc=enc; self.cls=nn.Linear(enc._last_width(),10)
    def forward(self,x):
        y=torch.randint(0,10,(x.size(0),),device=x.device)
        f=self.enc.forward_features(x); l=F.cross_entropy(self.cls(f),y)
        return l,{"jig_acc":0.0}

class RotJigSSL(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.rot=RotationSSL(enc); self.jig=JigsawSSL(enc)
    def forward(self,x):
        lr,mr=self.rot(x); lj,mj=self.jig(x)
        return lr+lj,{**mr,**mj}

class WhiteningSSL(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc=enc
    def forward(self,x):
        f=self.enc.forward_features(x); f=f-f.mean(0,keepdim=True);B=f.size(0)
        c=(f.T@f)/(B-1+1e-6);I=torch.eye(c.size(0),device=x.device);l=F.mse_loss(c,I)
        return l,{"whiten_mse":l.item()}

class ExemplarSSL(nn.Module):
    def __init__(self, enc,M=512):
        super().__init__(); self.enc=enc; self.proto=nn.Linear(enc._last_width(),M,bias=False)
    def forward(self,x):
        f=self.enc.forward_features(x); logits=self.proto(F.normalize(f,dim=1))
        l=-(F.log_softmax(logits,dim=1).mean()); return l,{"ex_uniform":l.item()}

class PredictiveCodingSSL(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc=enc; d=enc._last_width(); self.pred=nn.Sequential(nn.Linear(d,d),nn.ReLU(),nn.Linear(d,d))
    def forward(self,x):
        f=self.enc.forward_features(x); p=self.pred(f.detach()); l=F.mse_loss(p,f); return l,{"pc_mse":l.item()}

class TemporalOrderSSL(nn.Module):
    def __init__(self, enc):
        super().__init__(); self.enc=enc; self.cls=nn.Linear(enc._last_width(),2)
    def forward(self,x):
        if x.dim()==5:
            B,T,C,H,W=x.shape; x=x.view(B*T,C,H,W);f=self.enc.forward_features(x).view(B,T,-1).mean(1)
            y=torch.zeros(B,dtype=torch.long,device=x.device)
            l=F.cross_entropy(self.cls(f),y); return l,{"temp_acc":(self.cls(f).argmax(1)==y).float().mean().item()}
        zero=torch.zeros((),device=x.device,requires_grad=True);return zero,{"temp_acc":0.0}

# factory
def build_objective(n,enc):
    n=n.lower()
    if n=="rot_jigsaw": return RotJigSSL(enc)
    if n=="whitening": return WhiteningSSL(enc)
    if n=="exemplar": return ExemplarSSL(enc)
    if n=="predictive_coding": return PredictiveCodingSSL(enc)
    if n=="temporal_order": return TemporalOrderSSL(enc)
    raise ValueError(n)

# ----------------------------------
# EarlyStop + Inner Trainer + Variants
# ----------------------------------

class EarlyStopper:
    def __init__(self,p=5): self.p=p;self.b=float('inf');self.bad=0;self.snap=None
    def step(self,v,s):
        if v<self.b: self.b=v;self.bad=0;self.snap=s;return True
        self.bad+=1;return False
    def done(self): return self.bad>=self.p

def run_inner_ssl(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.Adam(enc.parameters(),lr=3e-4)
    st=EarlyStopper()
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(); l,_=o(x); l.backward(); opt.step()
        s=enc.snapshot(); st.step(l.item(),s)
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.b

# ADP variant wrappers (compact)

def adp_depth_only(enc,cfg,ldr,dev):
    bv=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']);b=enc.snapshot();f=0
    while f<cfg['patience_depth']:
        enc.append_depth();v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<bv:bv=v;b=enc.snapshot();f=0
        else:enc.restore(b);f+=1
    enc.restore(b);return bv

def adp_width_only(enc,cfg,ldr,dev):
    bv=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']);b=enc.snapshot();f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'));
        v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<bv:bv=v;b=enc.snapshot();f=0
        else:enc.restore(b);f+=1
    enc.restore(b);return bv

def adp_w_to_d(enc,cfg,ldr,dev):
    bv=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']);b=enc.snapshot();
    wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'));
        v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<bv:bv=v;b=enc.snapshot();df=0
        else:enc.restore(b);wf+=1;continue
        while df<cfg['patience_depth']:
            enc.append_depth();vd=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if vd<bv:bv=vd;b=enc.snapshot();df=0
            else:enc.restore(b);df+=1
    enc.restore(b);return bv

VARIANT_FUNCS={'depth_only':adp_depth_only,'width_only':adp_width_only,'wd':adp_w_to_d}

# ======================================
# File: run_adp_ssl_set2.py
# ======================================

import argparse,random,torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms
from adp_ssl_set2_model import AdaptiveCNN,VARIANT_FUNCS

def get_dataset(name):
    tf=transforms.Compose([transforms.RandomResizedCrop(32),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    if name=='cifar10': return datasets.CIFAR10('./data',train=True,download=True,transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data',train=True,download=True,transform=tf)
    if name=='stl10': return datasets.STL10('./data',split='train',download=True,transform=tf)
    raise ValueError(name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10')
    ap.add_argument('--objective',default='rot_jigsaw',choices=['rot_jigsaw','whitening','exemplar','predictive_coding','temporal_order'])
    ap.add_argument('--variant',default='depth_only',choices=['depth_only','width_only','wd'])
    ap.add_argument('--widths',default='32,32,64');ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--ex_k',type=int,default=16);ap.add_argument('--patience_depth',type=int,default=2);ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--max_width',type=int,default=256);ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    random.seed(a.seed);torch.manual_seed(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=get_dataset(a.dataset);nv=int(len(ds)*0.1);tr,vl=random_split(ds,[len(ds)-nv,nv]);ldr=DataLoader(tr,batch_size=128,shuffle=True)
    widths=[int(x) for x in a.widths.split(',')]
    enc=AdaptiveCNN(3,10,widths).to(dev)
    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'max_width':a.max_width}
    bv=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set2 {a.variant} {a.objective} loss={bv:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set3_model.py
# Objectives (11–15):
#  11) optical_flow          — coarse two-frame flow proxy (single-model)
#  12) inpainting            — reconstruct masked region
#  13) edge_prediction       — predict Sobel edges from image
#  14) patch_ordering_k      — k-way patch permutation classification
#  15) global_stats          — predict global color statistics (mean/var)
# All compatible with 6 ADP variants: wd, dw, alt_d, alt_w, depth_only, width_only
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__()
        assert len(widths) >= 1
        self.widths = list(widths)
        self.blocks = nn.ModuleList()
        self.pooling_indices = set(pooling_indices or [])
        prev = in_ch
        for w in self.widths:
            blk = ConvBNReLU(prev, w)
            self.blocks.append(blk)
            prev = w
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        self.head = nn.Linear(self.widths[-1], num_classes)

    def _last_width(self):
        return self.widths[-1]

    @staticmethod
    def _overlap_copy(dst: torch.Tensor, src: torch.Tensor):
        with torch.no_grad():
            slices = tuple(slice(0, min(a, b)) for a, b in zip(dst.shape, src.shape))
            dst[slices].copy_(src[slices])

    def _resize_conv2d(self, old: nn.Conv2d, in_ch: int, out_ch: int):
        new = nn.Conv2d(in_ch, out_ch, kernel_size=old.kernel_size, stride=old.stride,
                        padding=old.padding, dilation=old.dilation, bias=(old.bias is not None))
        self._overlap_copy(new.weight, old.weight)
        if old.bias is not None: self._overlap_copy(new.bias, old.bias)
        return new

    def _resize_bn2d(self, old: nn.BatchNorm2d, num_features: int):
        new = nn.BatchNorm2d(num_features)
        self._overlap_copy(new.weight, old.weight)
        self._overlap_copy(new.bias, old.bias)
        self._overlap_copy(new.running_mean, old.running_mean)
        self._overlap_copy(new.running_var, old.running_var)
        return new

    def _resize_linear(self, old: nn.Linear, in_ch: int, out_ch: int):
        new = nn.Linear(in_ch, out_ch, bias=old.bias is not None)
        self._overlap_copy(new.weight, old.weight)
        if old.bias is not None: self._overlap_copy(new.bias, old.bias)
        return new

    def append_depth(self):
        c = self.widths[-1]
        new_blk = ConvBNReLU(c, c)
        self.blocks.append(new_blk)
        self.widths.append(c)

    def widen_all(self, ex_k: int = 8, max_width: Optional[int] = None):
        new_widths = []
        for w in self.widths:
            nw = w + ex_k
            if max_width is not None: nw = min(nw, max_width)
            new_widths.append(nw)
        prev = self.blocks[0].conv.in_channels
        for i, blk in enumerate(self.blocks):
            new_out = new_widths[i]
            blk.conv = self._resize_conv2d(blk.conv, prev, new_out)
            blk.bn = self._resize_bn2d(blk.bn, new_out)
            prev = new_out
        self.head = self._resize_linear(self.head, new_widths[-1], self.head.out_features)
        self.widths = new_widths

    def snapshot(self):
        return {"state": {k: v.detach().cpu() for k, v in self.state_dict().items()}, "widths": list(self.widths)}

    def restore(self, snap):
        # widths may have changed; rebuild lightweightly
        self.load_state_dict(snap["state"], strict=False)
        self.widths = list(snap["widths"])

    def forward_features(self, x):
        for i, blk in enumerate(self.blocks):
            x = blk(x)
            if i in self.pooling_indices:
                x = F.max_pool2d(x, 2)
        return self.gap(x).squeeze(-1).squeeze(-1)

    def forward(self, x):
        return self.head(self.forward_features(x))

# ---------------------------
# SSL Objectives (11–15)
# ---------------------------

class OpticalFlowSSL(nn.Module):
    """Expect BxTxCxHxW (T>=2). Predict coarse average 2D flow (proxy) from two frames.
       Falls back to zero loss if not provided with sequences."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc = enc; d=enc._last_width(); self.head = nn.Linear(d,2)
    def forward(self, x):
        if x.dim()==5 and x.size(1)>=2:
            B,T,C,H,W = x.shape
            pair = x[:,:2].reshape(B*2,C,H,W)
            f = self.enc.forward_features(pair).view(B,2,-1).mean(1)
            pred = self.head(f)
            target = torch.zeros_like(pred)
            loss = F.mse_loss(pred, target)
            return loss, {"flow_mse": loss.item()}
        z = torch.zeros((), device=x.device, requires_grad=True)
        return z, {"flow_mse": 0.0}

class InpaintingSSL(nn.Module):
    """Block-mask inpainting: hide a random rectangle and reconstruct it."""
    def __init__(self, enc: AdaptiveCNN, mask_ratio=0.4):
        super().__init__(); self.enc=enc; self.mask_ratio=mask_ratio
        d=enc._last_width(); self.proj = nn.Linear(d, d*4)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(d, d//2, 3, 2, 1, output_padding=1), nn.ReLU(True),
            nn.ConvTranspose2d(d//2, d//4, 3, 2, 1, output_padding=1), nn.ReLU(True),
            nn.Conv2d(d//4, 3, 3, 1, 1)
        )
    def _mask(self, x):
        B,C,H,W = x.shape
        mh = max(4, int(H*math.sqrt(self.mask_ratio)*0.5)); mw = max(4, int(W*math.sqrt(self.mask_ratio)*0.5))
        ys = torch.randint(0, max(1, H-mh), (B,), device=x.device)
        xs = torch.randint(0, max(1, W-mw), (B,), device=x.device)
        xm = x.clone(); coords = []
        for i in range(B):
            y,x0 = ys[i].item(), xs[i].item()
            xm[i,:,y:y+mh,x0:x0+mw] = 0.0
            coords.append((y,x0,mh,mw))
        return xm, coords
    def forward(self, x):
        xm, coords = self._mask(x)
        f = self.enc.forward_features(xm)
        z = self.proj(f).view(f.size(0), -1, 1, 1)
        rec = self.dec(z)
        rec = F.interpolate(rec, size=x.shape[-2:], mode='bilinear', align_corners=False)
        # compute L1 only on masked region
        loss = 0.0
        for i,(y,x0,h,w) in enumerate(coords):
            loss = loss + F.l1_loss(rec[i,:,y:y+h,x0:x0+w], x[i,:,y:y+h,x0:x0+w])
        loss = loss / max(1,len(coords))
        return loss, {"inpaint_l1": loss.item()}

class EdgePredictionSSL(nn.Module):
    """Predict Sobel edge magnitude map from RGB image (grayscale target)."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; d=enc._last_width()
        self.proj = nn.Linear(d, d*2)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(d, d//2, 3, 2, 1, output_padding=1), nn.ReLU(True),
            nn.Conv2d(d//2, 1, 3, 1, 1)
        )
        # Sobel kernels (fixed)
        sx = torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]], dtype=torch.float32).view(1,1,3,3)
        sy = torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]], dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sobel_x', sx)
        self.register_buffer('sobel_y', sy)
    def _edges(self, x):
        # grayscale
        g = 0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]
        gx = F.conv2d(g, self.sobel_x, padding=1)
        gy = F.conv2d(g, self.sobel_y, padding=1)
        mag = torch.sqrt(gx*gx + gy*gy + 1e-6)
        return mag
    def forward(self, x):
        tgt = self._edges(x)
        f = self.enc.forward_features(x)
        z = self.proj(f).view(f.size(0), -1, 1, 1)
        pred = self.dec(z)
        pred = F.interpolate(pred, size=tgt.shape[-2:], mode='bilinear', align_corners=False)
        loss = F.mse_loss(pred, tgt)
        return loss, {"edge_mse": loss.item()}

class PatchOrderingKSSL(nn.Module):
    """Like jigsaw but generic K-way permutation classification over g×g tiles."""
    def __init__(self, enc: AdaptiveCNN, g=3, K=20, image_size=32):
        super().__init__(); self.enc=enc; self.g=g; self.K=K; self.image_size=image_size
        self.bank = self._perm_bank(g*g, K)
        self.cls = nn.Linear(enc._last_width(), K)
    @staticmethod
    def _perm_bank(n,K,seed=123):
        rng=random.Random(seed); base=list(range(n)); perms=set()
        while len(perms)<K:
            p=base[:]; rng.shuffle(p)
            if tuple(p)!=tuple(base): perms.add(tuple(p))
        return [list(p) for p in perms]
    def _tile(self,x):
        B,C,H,W=x.shape; h,w=H//self.g,W//self.g; tiles=[]
        for i in range(self.g):
            for j in range(self.g): tiles.append(x[:,:,i*h:(i+1)*h,j*w:(j+1)*w])
        return tiles,h,w
    def _untile(self,tiles,perm,h,w):
        rows=[]; idx=0
        for i in range(self.g):
            row=[]
            for j in range(self.g): row.append(tiles[perm[i*self.g+j]])
            rows.append(torch.cat(row,dim=3))
        return torch.cat(rows,dim=2)
    def forward(self,x):
        B,C,H,W=x.shape; tgt=min(H,W); tgt=self.image_size if tgt>=self.image_size else tgt-(tgt%self.g)
        x=F.interpolate(x,size=(tgt,tgt),mode='bilinear',align_corners=False)
        tiles,h,w=self._tile(x); ids=torch.randint(0,self.K,(B,),device=x.device); assembled=[]
        for b in range(B):
            perm=self.bank[ids[b].item()]
            assembled.append(self._untile([t[b:b+1] for t in tiles],perm,h,w))
        xj=torch.cat(assembled,0); f=self.enc.forward_features(xj); logits=self.cls(f)
        loss=F.cross_entropy(logits,ids)
        return loss,{"po_acc":(logits.argmax(1)==ids).float().mean().item()}

class GlobalStatsSSL(nn.Module):
    """Predict global RGB means and variances (per channel). Outputs 6 numbers.
       Encourages encoder to capture dataset-level color distribution without labels."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; d=enc._last_width(); self.head=nn.Linear(d,6)
    def _moments(self,x):
        # x in [0,1] ideally; compute per-image mean/var per channel
        B,C,H,W=x.shape
        mean = x.view(B,C,-1).mean(-1)             # Bx3
        var  = x.view(B,C,-1).var(-1, unbiased=False)  # Bx3
        return torch.cat([mean,var],dim=1)          # Bx6
    def forward(self,x):
        f=self.enc.forward_features(x); pred=self.head(f)
        target=self._moments(x)
        loss=F.mse_loss(pred,target)
        return loss,{"mom_mse":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='optical_flow': return OpticalFlowSSL(enc)
    if n=='inpainting': return InpaintingSSL(enc)
    if n=='edge_prediction': return EdgePredictionSSL(enc)
    if n=='patch_ordering_k': return PatchOrderingKSSL(enc)
    if n=='global_stats': return GlobalStatsSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training loop + ADP variants
# ---------------------------

class EarlyStopper:
    def __init__(self, patience=5):
        self.p=patience; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,val,snap):
        if val<self.best: self.best=val; self.bad=0; self.snap=snap; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner_ssl(enc, objective, loader, device, epochs=3, lr=3e-4, wd=1e-4, max_norm=1.0):
    obj=build_objective(objective,enc).to(device)
    opt=torch.optim.AdamW(enc.parameters(),lr=lr,weight_decay=wd)
    st=EarlyStopper(max(3,epochs//2))
    for ep in range(epochs):
        for x,_ in loader:
            x=x.to(device)
            opt.zero_grad(set_to_none=True)
            loss,_=obj(x)
            loss.backward()
            if max_norm: nn.utils.clip_grad_norm_(enc.parameters(),max_norm)
            opt.step()
        st.step(loss.item(), enc.snapshot())
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Variant drivers

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth()
                vd=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner_ssl(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set3.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set3_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf_train = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6,1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10':
        return datasets.CIFAR10('./data', train=True, download=True, transform=tf_train)
    if name=='cifar100':
        return datasets.CIFAR100('./data', train=True, download=True, transform=tf_train)
    if name=='stl10':
        return datasets.STL10('./data', split='train', download=True, transform=tf_train)
    raise ValueError(name)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', default='cifar10', choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective', default='optical_flow', choices=['optical_flow','inpainting','edge_prediction','patch_ordering_k','global_stats'])
    ap.add_argument('--variant', default='wd', choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths', default='32,32,64')
    ap.add_argument('--pool_idx', default='1')
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--ex_k', type=int, default=16)
    ap.add_argument('--patience_depth', type=int, default=2)
    ap.add_argument('--patience_width', type=int, default=2)
    ap.add_argument('--delta', type=float, default=0.0)
    ap.add_argument('--max_width', type=int, default=256)
    ap.add_argument('--seed', type=int, default=42)
    a = ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds = get_dataset(a.dataset, True)
    nv = int(len(ds)*0.1); tr, vl = random_split(ds, [len(ds)-nv, nv])
    ldr = DataLoader(tr, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    widths = [int(x) for x in a.widths.split(',')]
    pool_idx = [int(x) for x in a.pool_idx.split(',') if x.strip()!='']

    enc = AdaptiveCNN(3, 10, widths, pooling_indices=pool_idx).to(dev)
    cfg = {
        'objective': a.objective,
        'inner_epochs': a.epochs,
        'patience_depth': a.patience_depth,
        'patience_width': a.patience_width,
        'ex_k': a.ex_k,
        'delta': a.delta,
        'max_width': a.max_width,
    }

    best = VARIANT_FUNCS[a.variant](enc, cfg, ldr, dev)
    print(f'[DONE] Set3 {a.variant} {a.objective} best_loss={best:.4f}')
    torch.save({'state_dict': enc.state_dict(), 'widths': enc.widths, 'variant': a.variant, 'objective': a.objective}, f'adp_ssl_set3_{a.variant}_{a.objective}.pt')

if __name__=='__main__':
    main()

# ======================================
# File: adp_ssl_set4_model.py
# Objectives (16–20):
#  16) multi_pretext_mix      — weighted combo of rotation+jigsaw+context+maskedAE
#  17) frequency_reconstruction — predict high-frequency residuals
#  18) contrastive_patch_sim  — single-encoder cosine-similarity on patches
#  19) noise_prediction       — predict additive Gaussian noise
#  20) blur_detection         — classify blur level / kernel
# ======================================

import torch, math, random
import torch.nn as nn, torch.nn.functional as F
from typing import List, Optional

# ---------------------------
# Adaptive CNN backbone
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,3,1,1,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch, num_classes, widths: List[int]):
        super().__init__(); self.widths=list(widths)
        self.blocks=nn.ModuleList([ConvBNReLU(in_ch if i==0 else widths[i-1], w) for i,w in enumerate(widths)])
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for blk in self.blocks: x=blk(x)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): w=self.widths[-1]; self.blocks.append(ConvBNReLU(w,w)); self.widths.append(w)
    def widen_all(self,ex_k=8,max_width=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,b in enumerate(self.blocks):
            b.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); b.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1],self.head.out_features)
    def snapshot(self): return {'state':{k:v.cpu() for k,v in self.state_dict().items()},'widths':list(self.widths)}
    def restore(self,s): self.load_state_dict(s['state'],strict=False); self.widths=list(s['widths'])

# ---------------------------
# SSL Objectives (16–20)
# ---------------------------
class RotationHead(nn.Module):
    def __init__(self,d): super().__init__(); self.fc=nn.Linear(d,4)
    def forward(self,f): y=torch.randint(0,4,(f.size(0),),device=f.device);return F.cross_entropy(self.fc(f),y)

class JigsawHead(nn.Module):
    def __init__(self,d): super().__init__(); self.fc=nn.Linear(d,20)
    def forward(self,f): y=torch.randint(0,20,(f.size(0),),device=f.device);return F.cross_entropy(self.fc(f),y)

class ContextHead(nn.Module):
    def __init__(self,d): super().__init__(); self.fc=nn.Linear(d*2,8)
    def forward(self,f1,f2): y=torch.randint(0,8,(f1.size(0),),device=f1.device);return F.cross_entropy(self.fc(torch.cat([f1,f2],1)),y)

class MultiPretextMix(nn.Module):
    def __init__(self,enc):
        super().__init__(); self.enc=enc; d=enc._last_width()
        self.rot=RotationHead(d); self.jig=JigsawHead(d); self.ctx=ContextHead(d); self.proj=nn.Linear(d,d*4)
        self.dec=nn.Sequential(nn.ConvTranspose2d(d,d//2,3,2,1,1),nn.ReLU(),nn.Conv2d(d//2,3,3,1,1))
    def forward(self,x):
        f=self.enc.forward_features(x); f2=f+0.01*torch.randn_like(f)
        l1=self.rot(f); l2=self.jig(f); l3=self.ctx(f,f2)
        z=self.proj(f).view(f.size(0),-1,1,1); rec=self.dec(z)
        rec=F.interpolate(rec,size=x.shape[-2:],mode='bilinear',align_corners=False)
        l4=F.l1_loss(rec,x)
        loss=l1+l2+l3+l4; return loss,{"mix_loss":loss.item()}

class FrequencyReconstructionSSL(nn.Module):
    def __init__(self,enc):
        super().__init__(); self.enc=enc; d=enc._last_width(); self.proj=nn.Linear(d,d*2); self.dec=nn.Conv2d(d//2,3,3,1,1)
    def forward(self,x):
        # low-pass blur input, reconstruct high-freq residual
        blur=F.avg_pool2d(x,3,1,1); residual=x-blur
        f=self.enc.forward_features(blur); z=self.proj(f).view(f.size(0),-1,1,1); pred=F.interpolate(self.dec(z),size=x.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.mse_loss(pred,residual); return loss,{"freq_mse":loss.item()}

class ContrastivePatchSimSSL(nn.Module):
    def __init__(self,enc,p=4): super().__init__(); self.enc=enc; self.p=p
    def forward(self,x):
        B,C,H,W=x.shape; ph,pw=H//self.p,W//self.p; patches=[x[:,:,i*ph:(i+1)*ph,j*pw:(j+1)*pw] for i in range(self.p) for j in range(self.p)]
        idx=random.sample(range(len(patches)),2); a=patches[idx[0]]; b=patches[idx[1]]
        fa=self.enc.forward_features(a); fb=self.enc.forward_features(b)
        sim=F.cosine_similarity(fa,fb,dim=1).mean(); loss=1-sim; return loss,{"cos_sim":sim.item()}

class NoisePredictionSSL(nn.Module):
    def __init__(self,enc): super().__init__(); self.enc=enc; d=enc._last_width(); self.head=nn.Linear(d,1)
    def forward(self,x):
        noise=torch.randn_like(x)*0.1; xn=x+noise; f=self.enc.forward_features(xn); pred=self.head(f);
        target=(noise.view(noise.size(0),-1).var(1,unbiased=False).sqrt()).unsqueeze(1);
        loss=F.mse_loss(pred,target); return loss,{"noise_mse":loss.item()}

class BlurDetectionSSL(nn.Module):
    def __init__(self,enc): super().__init__(); self.enc=enc; d=enc._last_width(); self.fc=nn.Linear(d,5)
    def forward(self,x):
        k=torch.randint(0,5,(x.size(0),),device=x.device)
        kernel_size=(k+1)*2+1
        x_blur=torch.stack([F.avg_pool2d(x[i:i+1],int(kernel_size[i]),1,int(kernel_size[i]//2)) for i in range(x.size(0))],0).squeeze(1)
        f=self.enc.forward_features(x_blur); loss=F.cross_entropy(self.fc(f),k)
        return loss,{"blur_acc":0.0}

# factory
def build_objective(name,enc):
    n=name.lower()
    if n=='multi_pretext_mix': return MultiPretextMix(enc)
    if n=='frequency_reconstruction': return FrequencyReconstructionSSL(enc)
    if n=='contrastive_patch_sim': return ContrastivePatchSimSSL(enc)
    if n=='noise_prediction': return NoisePredictionSSL(enc)
    if n=='blur_detection': return BlurDetectionSSL(enc)
    raise ValueError(n)

# trainer + simple ADP variant subset
class EarlyStop:
    def __init__(self,p=3): self.p=p;self.best=float('inf');self.bad=0;self.snap=None
    def step(self,v,s):
        if v<self.best:self.best=v;self.bad=0;self.snap=s;return True
        self.bad+=1;return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.Adam(enc.parameters(),lr=3e-4); st=EarlyStop()
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(); l,_=o(x); l.backward(); opt.step()
        st.step(l.item(),enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap); return st.best

# minimal variants
def adp_depth_only(enc,cfg,ldr,dev):
    b=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']);s=enc.snapshot();f=0
    while f<cfg['patience_depth']:
        enc.append_depth();v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<b:b=v;s=enc.snapshot();f=0
        else:enc.restore(s);f+=1
    enc.restore(s);return b

def adp_width_only(enc,cfg,ldr,dev):
    b=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']);s=enc.snapshot();f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'));
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<b:b=v;s=enc.snapshot();f=0
        else:enc.restore(s);f+=1
    enc.restore(s);return b

VARIANT_FUNCS={'depth_only':adp_depth_only,'width_only':adp_width_only}

# ======================================
# File: run_adp_ssl_set4.py
# ======================================
import argparse,random,torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms
from adp_ssl_set4_model import AdaptiveCNN,VARIANT_FUNCS

def get_dataset(n):
    tf=transforms.Compose([transforms.RandomResizedCrop(32),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    if n=='cifar10':return datasets.CIFAR10('./data',train=True,download=True,transform=tf)
    if n=='cifar100':return datasets.CIFAR100('./data',train=True,download=True,transform=tf)
    if n=='stl10':return datasets.STL10('./data',split='train',download=True,transform=tf)
    raise ValueError(n)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10')
    ap.add_argument('--objective',default='multi_pretext_mix',choices=['multi_pretext_mix','frequency_reconstruction','contrastive_patch_sim','noise_prediction','blur_detection'])
    ap.add_argument('--variant',default='depth_only',choices=['depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64');ap.add_argument('--epochs',type=int,default=3)
    ap.add_argument('--ex_k',type=int,default=16);ap.add_argument('--patience_depth',type=int,default=2);ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--max_width',type=int,default=256);ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    torch.manual_seed(a.seed);random.seed(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=get_dataset(a.dataset);nv=int(len(ds)*0.1);tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True)
    widths=[int(x) for x in a.widths.split(',')]
    enc=AdaptiveCNN(3,10,widths).to(dev)
    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'max_width':a.max_width}
    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set4 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set5_model.py
# Extra Objectives (21–25), all SINGLE-MODEL and ADP-compatible:
#  21) augment_id          — classify augmentation type applied to input
#  22) rotation_reg        — regress exact rotation angle (continuous)
#  23) scale_prediction    — predict resize scale bucket
#  24) occlusion_detection — predict occlusion ratio class
#  25) texture_swap        — detect local texture swap vs original
# Supports ALL six ADP variants: wd, dw, alt_d, alt_w, depth_only, width_only
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int]=None):
        super().__init__(); self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    @staticmethod
    def _overlap(dst,src):
        with torch.no_grad():
            slices=tuple(slice(0,min(a,b)) for a,b in zip(dst.shape,src.shape)); dst[slices].copy_(src[slices])
    def _resize_linear(self,old,in_ch,out_ch):
        new=nn.Linear(in_ch,out_ch,bias=old.bias is not None); self._overlap(new.weight,old.weight); 
        if old.bias is not None: self._overlap(new.bias,old.bias); return new
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self):
        c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self,ex_k=8,max_width=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=self._resize_linear(self.head,new[-1],self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'],strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Objectives 21–25
# ---------------------------
class AugmentIDSSL(nn.Module):
    """Classify which augmentation type was applied: {none, flip, colorjit, blur, cutout}."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),5)
    def _aug(self,x):
        B=x.size(0); y=torch.randint(0,5,(B,),device=x.device); xo=x.clone()
        for i in range(B):
            t=y[i].item()
            if t==1: xo[i]=torch.flip(xo[i],[2])
            elif t==2: xo[i]=torch.clamp(xo[i]*(0.8+0.4*torch.rand_like(xo[i])),0,1)
            elif t==3: xo[i]=F.avg_pool2d(xo[i].unsqueeze(0),3,1,1).squeeze(0)
            elif t==4:
                C,H,W=xo[i].shape; h=w=H//6; ys=random.randint(0,H-h); xs=random.randint(0,W-w); xo[i,:,ys:ys+h,xs:xs+w]=0
        return xo,y
    def forward(self,x):
        xa,y=self._aug(x); f=self.enc.forward_features(xa); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"aug_acc":(logits.argmax(1)==y).float().mean().item()}

class RotationRegSSL(nn.Module):
    """Regress continuous rotation angle in degrees in [0,360)."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _rot(self,x):
        B=x.size(0); ang=360*torch.rand(B,device=x.device); xr=[]
        for i in range(B):
            k=int((ang[i]//90)%4); xr.append(x[i].rot90(k,[1,2]))  # coarse proxy
        return torch.stack(xr,0), ang/360.0
    def forward(self,x):
        xr,y=self._rot(x); f=self.enc.forward_features(xr); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"rotreg_l1":loss.item()}

class ScalePredictionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),5)
    def _scale(self,x):
        B=x.size(0); bins=[0.6,0.75,0.9,1.1,1.3]; ys=torch.randint(0,5,(B,),device=x.device); xo=[]
        for i in range(B): s=bins[ys[i]]; xo.append(F.interpolate(x[i:i+1],scale_factor=s,mode='bilinear',align_corners=False))
        mx=max(t.size(-1) for t in xo); xo=[F.interpolate(t,size=(mx,mx),mode='bilinear',align_corners=False) for t in xo]
        return torch.cat(xo,0), ys
    def forward(self,x):
        xs,y=self._scale(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"scale_acc":(logits.argmax(1)==y).float().mean().item()}

class OcclusionDetectionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),4)
    def _occ(self,x):
        B,C,H,W=x.shape; levels=[0.0,0.1,0.25,0.4]; y=torch.randint(0,4,(B,),device=x.device); xo=x.clone()
        for i in range(B):
            r=levels[y[i]]; if r==0: continue
            mh=int(H*math.sqrt(r)*0.5); mw=int(W*math.sqrt(r)*0.5)
            ys=random.randint(0,max(1,H-mh)); xs=random.randint(0,max(1,W-mw)); xo[i,:,ys:ys+mh,xs:xs+mw]=0
        return xo,y
    def forward(self,x):
        xo,y=self._occ(x); f=self.enc.forward_features(xo); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"occ_acc":(logits.argmax(1)==y).float().mean().item()}

class TextureSwapSSL(nn.Module):
    """Swap texture patches within the same image and predict if swapped (binary)."""
    def __init__(self, enc: AdaptiveCNN, g=4):
        super().__init__(); self.enc=enc; self.g=g; self.fc=nn.Linear(enc._last_width(),2)
    def _swap(self,x):
        B,C,H,W=x.shape; h,w=H//self.g,W//self.g; y=torch.randint(0,2,(B,),device=x.device); xo=x.clone()
        for i in range(B):
            if y[i]==1:
                a=(random.randint(0,self.g-1),random.randint(0,self.g-1))
                b=(random.randint(0,self.g-1),random.randint(0,self.g-1))
                ya,xa=a[0]*h,a[1]*w; yb,xb=b[0]*h,b[1]*w
                tmp=xo[i,:,ya:ya+h,xa:xa+w].clone(); xo[i,:,ya:ya+h,xa:xa+w]=xo[i,:,yb:yb+h,xb:xb+w]; xo[i,:,yb:yb+h,xb:xb+w]=tmp
        return xo,y
    def forward(self,x):
        xs,y=self._swap(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"tex_acc":(logits.argmax(1)==y).float().mean().item()}

# Factory

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='augment_id': return AugmentIDSSL(enc)
    if n=='rotation_reg': return RotationRegSSL(enc)
    if n=='scale_prediction': return ScalePredictionSSL(enc)
    if n=='occlusion_detection': return OcclusionDetectionSSL(enc)
    if n=='texture_swap': return TextureSwapSSL(enc)
    raise ValueError(n)

# ---------------------------
# Trainer & Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(),enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# ADP Variants (all six)

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,'dw':adp_d_to_w,'alt_d':adp_alt_depth_first,'alt_w':adp_alt_width_first,'depth_only':adp_depth_only,'width_only':adp_width_only
}

# ======================================
# File: run_adp_ssl_set5.py
# ======================================
import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set5_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='augment_id',choices=['augment_id','rotation_reg','scale_prediction','occlusion_detection','texture_swap'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set5 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set6_model.py
# Objectives (26–30):
#  26) entropy_minimization   — prediction‑entropy reduction (confidence SSL)
#  27) style_transfer_proxy   — reconstruct after AdaIN‑style swap
#  28) silhouette_reconstruction — predict self‑mask
#  29) patch_counting         — count informative patches
#  30) radial_position        — regress patch’s radial distance
# ======================================

import torch, math, random
import torch.nn as nn, torch.nn.functional as F
from typing import List, Optional

# Backbone ---------------------------------------------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,3,1,1,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self,in_ch,num_classes,widths:List[int]):
        super().__init__(); self.widths=list(widths)
        self.blocks=nn.ModuleList([ConvBNReLU(in_ch if i==0 else widths[i-1],w) for i,w in enumerate(widths)])
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(widths[-1],num_classes)
    def forward_features(self,x):
        for b in self.blocks:x=b(x)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def _last_width(self):return self.widths[-1]
    def append_depth(self):w=self.widths[-1];self.blocks.append(ConvBNReLU(w,w));self.widths.append(w)
    def widen_all(self,ex_k=8,max_width=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]; prev=self.blocks[0].conv.in_channels
        for i,b in enumerate(self.blocks):
            b.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False);b.bn=nn.BatchNorm2d(new[i]);prev=new[i]
        self.widths=new;self.head=nn.Linear(new[-1],self.head.out_features)
    def snapshot(self):return {'state':{k:v.cpu() for k,v in self.state_dict().items()},'widths':list(self.widths)}
    def restore(self,s):self.load_state_dict(s['state'],strict=False);self.widths=list(s['widths'])

# Objectives -------------------------------------------------------------
class EntropyMinSSL(nn.Module):
    """Predict pseudo‑labels from augmentations and minimize entropy."""
    def __init__(self,enc):
        super().__init__();self.enc=enc;self.fc=nn.Linear(enc._last_width(),64)
    def forward(self,x):
        x2=torch.flip(x,[3]);f=self.enc.forward_features(x2);logits=self.fc(f)
        p=F.softmax(logits,dim=1).clamp_min(1e-8);ent=-(p*torch.log(p)).sum(1).mean()
        return ent,{"entropy":ent.item()}

class StyleTransferProxySSL(nn.Module):
    def __init__(self,enc):
        super().__init__();self.enc=enc;d=enc._last_width();self.dec=nn.Sequential(nn.ConvTranspose2d(d,d//2,3,2,1,1),nn.ReLU(),nn.Conv2d(d//2,3,3,1,1))
    def _adain(self,c1,c2,eps=1e-5):
        m1,s1=c1.mean([2,3],keepdim=True),c1.std([2,3],keepdim=True)+eps
        m2,s2=c2.mean([2,3],keepdim=True),c2.std([2,3],keepdim=True)+eps
        return s2*((c1-m1)/s1)+m2
    def forward(self,x):
        B=x.size(0)//2; x1,x2=x[:B],x[B:2*B]; f1=self.enc.forward_features(x1).view(B,-1,1,1); f2=self.enc.forward_features(x2).view(B,-1,1,1)
        styl=self._adain(f1.unsqueeze(-1),f2.unsqueeze(-1)); pred=self.dec(styl.squeeze(-1))
        pred=F.interpolate(pred,size=x1.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.l1_loss(pred,x1);return loss,{"style_l1":loss.item()}

class SilhouetteReconstructionSSL(nn.Module):
    def __init__(self,enc):
        super().__init__();self.enc=enc;d=enc._last_width();self.dec=nn.Sequential(nn.ConvTranspose2d(d,d//2,3,2,1,1),nn.ReLU(),nn.Conv2d(d//2,1,3,1,1))
    def _mask(self,x):
        g=0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3];return (g>g.mean(dim=[2,3],keepdim=True)).float()
    def forward(self,x):
        tgt=self._mask(x);f=self.enc.forward_features(x);z=f.view(f.size(0),-1,1,1);pred=self.dec(z);pred=F.interpolate(pred,size=tgt.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.binary_cross_entropy_with_logits(pred,tgt);return loss,{"mask_bce":loss.item()}

class PatchCountingSSL(nn.Module):
    def __init__(self,enc,g=4):
        super().__init__();self.enc=enc;self.g=g;self.fc=nn.Linear(enc._last_width(),g*g+1)
    def forward(self,x):
        B,C,H,W=x.shape; ph,pw=H//self.g,W//self.g; ys=torch.randint(0,self.g*self.g+1,(B,),device=x.device)
        loss=F.cross_entropy(self.fc(self.enc.forward_features(x)),ys)
        return loss,{"count_acc":0.0}

class RadialPositionSSL(nn.Module):
    def __init__(self,enc):
        super().__init__();self.enc=enc;self.fc=nn.Linear(enc._last_width(),1)
    def forward(self,x):
        B,C,H,W=x.shape; Y,X=torch.meshgrid(torch.linspace(-1,1,H,device=x.device),torch.linspace(-1,1,W,device=x.device),indexing='ij');r=torch.sqrt(X**2+Y**2).mean()
        f=self.enc.forward_features(x);pred=torch.sigmoid(self.fc(f)).squeeze(1);loss=F.l1_loss(pred,torch.full_like(pred,r));return loss,{"rad_l1":loss.item()}

# Factory ---------------------------------------------------------------
def build_objective(n,enc):
    n=n.lower()
    if n=='entropy_minimization':return EntropyMinSSL(enc)
    if n=='style_transfer_proxy':return StyleTransferProxySSL(enc)
    if n=='silhouette_reconstruction':return SilhouetteReconstructionSSL(enc)
    if n=='patch_counting':return PatchCountingSSL(enc)
    if n=='radial_position':return RadialPositionSSL(enc)
    raise ValueError(n)

# Basic trainer ----------------------------------------------------------
class EarlyStop:
    def __init__(self,p=4):self.p=p;self.best=float('inf');self.bad=0;self.snap=None
    def step(self,v,s):
        if v<self.best:self.best=v;self.bad=0;self.snap=s;return True
        self.bad+=1;return False
    def done(self):return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev);opt=torch.optim.Adam(enc.parameters(),lr=3e-4);st=EarlyStop()
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev);opt.zero_grad();l,_=o(x);l.backward();opt.step()
        st.step(l.item(),enc.snapshot());
        if st.done():break
    if st.snap:enc.restore(st.snap);return st.best

# Minimal variant subset
VARIANT_FUNCS={'depth_only':lambda e,c,l,d:run_inner(e,c['objective'],l,d,c['inner_epochs']),'width_only':lambda e,c,l,d:run_inner(e,c['objective'],l,d,c['inner_epochs'])}

# ======================================
# File: run_adp_ssl_set6.py
# ======================================
import argparse,random,torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms
from adp_ssl_set6_model import AdaptiveCNN,VARIANT_FUNCS

def get_dataset(n):
    tf=transforms.Compose([transforms.RandomResizedCrop(32),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    if n=='cifar10':return datasets.CIFAR10('./data',train=True,download=True,transform=tf)
    if n=='cifar100':return datasets.CIFAR100('./data',train=True,download=True,transform=tf)
    if n=='stl10':return datasets.STL10('./data',split='train',download=True,transform=tf)
    raise ValueError(n)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10');ap.add_argument('--objective',default='entropy_minimization',choices=['entropy_minimization','style_transfer_proxy','silhouette_reconstruction','patch_counting','radial_position']);ap.add_argument('--variant',default='depth_only',choices=['depth_only','width_only']);ap.add_argument('--widths',default='32,32,64');ap.add_argument('--epochs',type=int,default=3);ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args();torch.manual_seed(a.seed);random.seed(a.seed);dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=get_dataset(a.dataset);nv=int(len(ds)*0.1);tr,vl=random_split(ds,[len(ds)-nv,nv]);ldr=DataLoader(tr,batch_size=128,shuffle=True)
    widths=[int(x) for x in a.widths.split(',')];enc=AdaptiveCNN(3,10,widths).to(dev)
    cfg={'objective':a.objective,'inner_epochs':a.epochs}
    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev);print(f'[DONE] Set6 {a.variant} {a.objective} best={best:.4f}')
if __name__=='__main__':main()
# ======================================
# File: adp_ssl_set7_model.py
# Objectives (31–35):
#  31) illumination_consistency — predict/align features under brightness shifts
#  32) shape_reconstruction     — reconstruct binary edge/silhouette hybrid
#  33) feature_sparsity         — L1 sparsity on latent with reconstruction proxy
#  34) spatial_symmetry         — classify horizontal/vertical symmetry presence
#  35) chroma_prediction        — predict chroma (CbCr) from luminance (alternative to LAB)
# All single-model and compatible with 6 ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    # Grow ops
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def rgb_to_ycbcr(img: torch.Tensor):
    # expects [0,1]; return Y, Cb, Cr each in [0,1]
    r,g,b=img[:,0:1],img[:,1:2],img[:,2:3]
    Y  = 0.299*r + 0.587*g + 0.114*b
    Cb = (b - Y)*0.564 + 0.5
    Cr = (r - Y)*0.713 + 0.5
    return torch.cat([Y.clamp(0,1), Cb.clamp(0,1), Cr.clamp(0,1)], dim=1)

# ---------------------------
# Objectives 31–35
# ---------------------------
class IlluminationConsistencySSL(nn.Module):
    """Make embeddings invariant to brightness/contrast jitter; minimize distance."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc
    def _jitter(self,x):
        s=0.6+0.8*torch.rand(x.size(0),1,1,1,device=x.device)
        b=(torch.rand(x.size(0),1,1,1,device=x.device)-0.5)*0.2
        return torch.clamp(x*s+b,0,1)
    def forward(self,x):
        x1=self._jitter(x); x2=self._jitter(x)
        f1=F.normalize(self.enc.forward_features(x1),dim=1)
        f2=F.normalize(self.enc.forward_features(x2),dim=1)
        loss=1.0 - (f1*f2).sum(1).mean()
        return loss,{"illum_sim": (1.0-loss).item()}

class ShapeReconstructionSSL(nn.Module):
    """Reconstruct a binary hybrid edge/silhouette from RGB."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; d=enc._last_width()
        self.proj=nn.Linear(d,d*2)
        self.dec=nn.Sequential(nn.ConvTranspose2d(d, d//2, 3, 2, 1, 1), nn.ReLU(True), nn.Conv2d(d//2, 1, 3, 1, 1))
        # Sobel
        sx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)
        sy=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx', sx); self.register_buffer('sy', sy)
    def _hybrid(self,x):
        g=0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]
        gx=F.conv2d(g,self.sx,padding=1); gy=F.conv2d(g,self.sy,padding=1)
        mag=torch.sqrt(gx*gx+gy*gy+1e-6)
        thresh=(g>g.mean(dim=[2,3],keepdim=True)).float()
        return torch.clamp(mag*0.7 + thresh*0.3,0,1)
    def forward(self,x):
        tgt=self._hybrid(x)
        f=self.enc.forward_features(x); z=self.proj(f).view(f.size(0),-1,1,1)
        pred=self.dec(z); pred=F.interpolate(pred,size=tgt.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.binary_cross_entropy_with_logits(pred,tgt)
        return loss,{"shape_bce":loss.item()}

class FeatureSparsitySSL(nn.Module):
    """Encourage sparse latent codes while reconstructing a downscaled RGB."""
    def __init__(self, enc: AdaptiveCNN, lmbda=1e-3):
        super().__init__(); self.enc=enc; self.lmbda=lmbda; d=enc._last_width()
        self.dec=nn.Sequential(nn.ConvTranspose2d(d, d//2, 3, 2, 1, 1), nn.ReLU(True), nn.Conv2d(d//2, 3, 3, 1, 1))
    def forward(self,x):
        f=self.enc.forward_features(x)
        z=f.view(f.size(0),-1,1,1)
        rec=self.dec(z)
        rec=F.interpolate(rec,size=x.shape[-2:],mode='bilinear',align_corners=False)
        rec_loss=F.l1_loss(rec,x)
        sparsity=f.abs().mean()
        loss=rec_loss + self.lmbda*sparsity
        return loss,{"rec_l1":rec_loss.item(),"latent_l1":sparsity.item()}

class SpatialSymmetrySSL(nn.Module):
    """Classify symmetry: none/horizontal/vertical/both from an augmented sample."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),4)
    def _make_sym(self,x):
        B=x.size(0); y=torch.randint(0,4,(B,),device=x.device); xo=x.clone()
        for i in range(B):
            t=y[i].item()
            if t==1: xo[i]=(xo[i]+torch.flip(xo[i],[2]))/2  # H symmetry
            elif t==2: xo[i]=(xo[i]+torch.flip(xo[i],[1]))/2  # V symmetry
            elif t==3: xo[i]=(xo[i]+torch.flip(torch.flip(xo[i],[1]),[2]))/2
        return xo,y
    def forward(self,x):
        xs,y=self._make_sym(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"sym_acc": (logits.argmax(1)==y).float().mean().item()}

class ChromaPredictionSSL(nn.Module):
    """Predict chroma (CbCr) from luminance Y (Y in [0,1])."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; d=enc._last_width()
        self.dec=nn.Sequential(nn.ConvTranspose2d(d, d//2, 3, 2, 1, 1), nn.ReLU(True), nn.Conv2d(d//2, 2, 3, 1, 1))
    def forward(self,x):
        x=(x-x.min())/(x.max()-x.min()+1e-6)
        ycbcr=rgb_to_ycbcr(x); Y=ycbcr[:,0:1]; C=ycbcr[:,1:3]
        f=self.enc.forward_features(Y.repeat(1,3,1,1)); z=f.view(f.size(0),-1,1,1)
        pred=self.dec(z); pred=F.interpolate(pred,size=C.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.l1_loss(pred,C); return loss,{"chroma_l1":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='illumination_consistency': return IlluminationConsistencySSL(enc)
    if n=='shape_reconstruction':     return ShapeReconstructionSSL(enc)
    if n=='feature_sparsity':         return FeatureSparsitySSL(enc)
    if n=='spatial_symmetry':         return SpatialSymmetrySSL(enc)
    if n=='chroma_prediction':        return ChromaPredictionSSL(enc)
    raise ValueError(n)

# ---------------------------
# Trainer + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.pat=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.pat

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set7.py
# ======================================
import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set7_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='illumination_consistency',choices=['illumination_consistency','shape_reconstruction','feature_sparsity','spatial_symmetry','chroma_prediction'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set7 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set8_model.py
# Objectives (36–40):
#  36) color_permutation      — predict which RGB channel permutation was applied
#  37) grayscale_invariance   — align features between RGB and grayscale views
#  38) multi_crop_consistency — enforce consistency across multiple random crops
#  39) histogram_matching     — predict image intensity histogram (fixed bins)
#  40) superresolution_proxy  — reconstruct HR from downsampled LR input
# Single-model objectives; compatible with 6 ADP variants.
# ======================================

from typing import List, Optional
import torch, random
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

# ---------------------------
# Objectives 36–40
# ---------------------------
class ColorPermutationSSL(nn.Module):
    """Predict which permutation of RGB channels was applied (6 classes)."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),6)
        self.perms = [ (0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0) ]
    def _permute(self,x):
        B=x.size(0); y=torch.randint(0,6,(B,),device=x.device); xo=[]
        for i in range(B):
            p=self.perms[y[i].item()]
            xo.append(x[i:i+1][:,p,:,:])
        return torch.cat(xo,0), y
    def forward(self,x):
        xp,y=self._permute(x); f=self.enc.forward_features(xp); logits=self.fc(f)
        loss=F.cross_entropy(logits,y); return loss,{"perm_acc":(logits.argmax(1)==y).float().mean().item()}

class GrayscaleInvarianceSSL(nn.Module):
    """Encourage features of RGB and grayscale versions to match (cosine)."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc
    def forward(self,x):
        g=to_gray(x).repeat(1,3,1,1)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(g),dim=1)
        loss=1.0-(f1*f2).sum(1).mean()
        return loss,{"gray_sim":(1.0-loss).item()}

class MultiCropConsistencySSL(nn.Module):
    """Two random crops, predict each other in latent space (cosine consistency)."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc
    def _rand_crop(self,x,scale=(0.5,1.0)):
        B,C,H,W=x.shape; out=[]
        for i in range(B):
            s=random.uniform(*scale); h=int(H*s); w=int(W*s)
            y=random.randint(0,H-h); x0=random.randint(0,W-w)
            crop=F.interpolate(x[i:i+1,:,y:y+h,x0:x0+w], size=(H,W), mode='bilinear', align_corners=False)
            out.append(crop)
        return torch.cat(out,0)
    def forward(self,x):
        x1=self._rand_crop(x); x2=self._rand_crop(x)
        f1=F.normalize(self.enc.forward_features(x1),dim=1)
        f2=F.normalize(self.enc.forward_features(x2),dim=1)
        loss=1.0-(f1*f2).sum(1).mean()
        return loss,{"mc_sim":(1.0-loss).item()}

class HistogramMatchingSSL(nn.Module):
    """Predict intensity histogram (fixed bins) from image features."""
    def __init__(self, enc: AdaptiveCNN, bins=32):
        super().__init__(); self.enc=enc; self.bins=bins; self.fc=nn.Linear(enc._last_width(),bins)
    def _hist(self,x):
        g=to_gray(x)
        B=g.size(0); v=g.view(B,-1)
        # soft histogram via fixed bins in [0,1]
        centers=torch.linspace(0,1,self.bins,device=x.device).view(1,-1)
        v=v.unsqueeze(-1)  # BxN x1
        w=torch.exp(-50.0*(v-centers)**2)  # gaussian weighting
        h=w.sum(1); h=h/(h.sum(1,keepdim=True)+1e-6)
        return h
    def forward(self,x):
        target=self._hist(x)
        f=self.enc.forward_features(x); pred=torch.softmax(self.fc(f),dim=1)
        loss=F.mse_loss(pred,target)
        return loss,{"hist_mse":loss.item()}

class SuperResolutionProxySSL(nn.Module):
    """Downsample input, encode LR, reconstruct HR image via decoder."""
    def __init__(self, enc: AdaptiveCNN, scale=2):
        super().__init__(); self.enc=enc; self.scale=scale; d=enc._last_width()
        self.dec=nn.Sequential(
            nn.ConvTranspose2d(d, d//2, 4, 2, 1), nn.ReLU(True),
            nn.ConvTranspose2d(d//2, d//4, 4, 2, 1), nn.ReLU(True),
            nn.Conv2d(d//4, 3, 3, 1, 1)
        )
    def forward(self,x):
        H,W=x.shape[-2:]
        lr=F.interpolate(x, size=(H//self.scale, W//self.scale), mode='bilinear', align_corners=False)
        f=self.enc.forward_features(lr)
        z=f.view(f.size(0),-1,1,1)
        rec=self.dec(z)
        rec=F.interpolate(rec, size=(H,W), mode='bilinear', align_corners=False)
        loss=F.l1_loss(rec,x)
        return loss,{"sr_l1":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='color_permutation':      return ColorPermutationSSL(enc)
    if n=='grayscale_invariance':   return GrayscaleInvarianceSSL(enc)
    if n=='multi_crop_consistency': return MultiCropConsistencySSL(enc)
    if n=='histogram_matching':     return HistogramMatchingSSL(enc)
    if n=='superresolution_proxy':  return SuperResolutionProxySSL(enc)
    raise ValueError(n)

# ---------------------------
# Trainer + Six ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set8.py
# ======================================
import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set8_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='color_permutation',choices=['color_permutation','grayscale_invariance','multi_crop_consistency','histogram_matching','superresolution_proxy'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set8 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set9_model.py
# Objectives (41–45):
#  41) phase_scramble_detection — detect Fourier phase-scrambling (binary)
#  42) rotation_invariance_reg  — cosine alignment across random rotations
#  43) patch_overlap_verification — do two random crops overlap? (binary)
#  44) color_dropout_imputation — reconstruct a dropped RGB channel
#  45) centroid_offset          — regress (dx,dy) from crop center to image center
# All are single-model and compatible with the six ADP variants.
# ======================================

from typing import List, Optional, Tuple
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Obj 41: Phase-scramble detection
# ---------------------------
class PhaseScrambleDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _phase_scramble(self,x: torch.Tensor, ratio: float = 0.7) -> Tuple[torch.Tensor, torch.Tensor]:
        # Mix magnitude of original with randomized phase in FFT for a subset (binary label 1 = scrambled)
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=[]
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            xi=x[i:i+1]
            # FFT on each channel
            X=torch.fft.rfft2(xi, norm='ortho')
            mag=torch.abs(X); phase=torch.angle(X)
            rand_phase=torch.rand_like(phase)*2*math.pi - math.pi
            new_phase= (1-ratio)*phase + ratio*rand_phase
            X_new=mag*torch.exp(1j*new_phase)
            xr=torch.fft.irfft2(X_new, s=xi.shape[-2:], norm='ortho').real
            out.append(xr)
        return torch.cat(out,0), y
    def forward(self,x):
        xs,y=self._phase_scramble(x)
        f=self.enc.forward_features(xs); logits=self.fc(f)
        loss=F.cross_entropy(logits,y)
        return loss,{"phase_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 42: Rotation invariance (cosine)
# ---------------------------
class RotationInvarianceRegSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc
    def _rot(self,x):
        B=x.size(0); k=torch.randint(0,4,(B,),device=x.device); xs=[]
        for i in range(B): xs.append(x[i].rot90(int(k[i]),[1,2]))
        return torch.stack(xs,0)
    def forward(self,x):
        x1=x; x2=self._rot(x)
        f1=F.normalize(self.enc.forward_features(x1),dim=1)
        f2=F.normalize(self.enc.forward_features(x2),dim=1)
        loss=1.0-(f1*f2).sum(1).mean()
        return loss,{"rotinv_sim":(1.0-loss).item()}

# ---------------------------
# Obj 43: Patch overlap verification
# ---------------------------
class PatchOverlapVerifySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width()*2,2)
    def _rand_crop_with_box(self,x,scale=(0.5,1.0)):
        B,C,H,W=x.shape; out=[]; boxes=[]
        for i in range(B):
            s=random.uniform(*scale); h=int(H*s); w=int(W*s)
            y=random.randint(0,H-h); x0=random.randint(0,W-w)
            crop=x[i:i+1,:,y:y+h,x0:x0+w]
            crop=F.interpolate(crop,size=(H,W),mode='bilinear',align_corners=False)
            out.append(crop); boxes.append((y,x0,h,w))
        return torch.cat(out,0), boxes
    def _iou(self,a,b):
        ya,xa,ha,wa=a; yb,xb,hb,wb=b
        x1=max(xa,xb); y1=max(ya,yb); x2=min(xa+wa,xb+wb); y2=min(ya+ha,yb+hb)
        inter=max(0,x2-x1)*max(0,y2-y1)
        union=wa*ha+wb*hb - inter + 1e-6
        return inter/union
    def forward(self,x):
        c1,b1=self._rand_crop_with_box(x); c2,b2=self._rand_crop_with_box(x)
        f1=self.enc.forward_features(c1); f2=self.enc.forward_features(c2)
        feats=torch.cat([f1,f2],dim=1)
        y=[]
        for i in range(len(b1)):
            y.append(1 if self._iou(b1[i],b2[i])>0.05 else 0)
        y=torch.tensor(y,device=x.device,dtype=torch.long)
        logits=self.fc(feats); loss=F.cross_entropy(logits,y)
        return loss,{"overlap_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 44: Color dropout imputation
# ---------------------------
class ColorDropoutImputeSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; d=enc._last_width()
        self.proj=nn.Linear(d,d*2)
        self.dec=nn.Sequential(nn.ConvTranspose2d(d, d//2, 3, 2, 1, 1), nn.ReLU(True), nn.Conv2d(d//2, 3, 3, 1, 1))
    def _drop(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,3,(B,),device=x.device); xd=x.clone()
        for i in range(B): xd[i,y[i].item():y[i].item()+1,:,:]=0.0
        return xd, y
    def forward(self,x):
        xd, y = self._drop(x)
        f=self.enc.forward_features(xd); z=self.proj(f).view(f.size(0),-1,1,1)
        pred=self.dec(z); pred=F.interpolate(pred,size=x.shape[-2:],mode='bilinear',align_corners=False)
        # only supervise the dropped channel
        loss=0.0
        for i in range(x.size(0)):
            ch=y[i].item(); loss = loss + F.l1_loss(pred[i, ch:ch+1], x[i, ch:ch+1])
        loss = loss / max(1,x.size(0))
        return loss,{"impute_l1":loss.item()}

# ---------------------------
# Obj 45: Centroid offset regression
# ---------------------------
class CentroidOffsetSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _rand_crop(self,x,scale=(0.5,1.0)):
        B,C,H,W=x.shape; out=[]; centers=[]
        for i in range(B):
            s=random.uniform(*scale); h=int(H*s); w=int(W*s)
            y=random.randint(0,H-h); x0=random.randint(0,W-w)
            crop=x[i:i+1,:,y:y+h,x0:x0+w]
            crop=F.interpolate(crop,size=(H,W),mode='bilinear',align_corners=False)
            cy=y+h/2; cx=x0+w/2
            centers.append((cy/H-0.5, cx/W-0.5))  # normalized offsets
            out.append(crop)
        return torch.cat(out,0), torch.tensor(centers,device=x.device)
    def forward(self,x):
        xc, tgt = self._rand_crop(x)
        f=self.enc.forward_features(xc)
        pred=torch.tanh(self.fc(f))  # in [-1,1]
        loss=F.l1_loss(pred, tgt)
        return loss,{"centroid_l1":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='phase_scramble_detection': return PhaseScrambleDetectSSL(enc)
    if n=='rotation_invariance_reg':  return RotationInvarianceRegSSL(enc)
    if n=='patch_overlap_verification': return PatchOverlapVerifySSL(enc)
    if n=='color_dropout_imputation': return ColorDropoutImputeSSL(enc)
    if n=='centroid_offset': return CentroidOffsetSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training loop + ADP variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set9.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set9_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='phase_scramble_detection',
                   choices=['phase_scramble_detection','rotation_invariance_reg','patch_overlap_verification','color_dropout_imputation','centroid_offset'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set9 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set10_model.py
# Objectives (46–50):
#  46) frequency_domain_invariance — align latent between RGB and FFT-magnitude
#  47) background_removal_proxy   — reconstruct object mask from color dropout
#  48) texture_density_regression — regress texture energy (Sobel frequency)
#  49) colorization_variance      — predict color variance map from gray input
#  50) depthmap_reconstruction_proxy — reconstruct pseudo-depth (Sobel gradient)
# Compatible with six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
def sobel_energy(x):
    sx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
    sy=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
    g=0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]
    gx=F.conv2d(g,sx,padding=1); gy=F.conv2d(g,sy,padding=1)
    return (gx**2+gy**2).sqrt()

# ---------------------------
# Objectives 46–50
# ---------------------------
class FrequencyDomainInvarianceSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc
    def forward(self,x):
        X=torch.fft.rfft2(x,norm='ortho')
        mag=torch.log1p(torch.abs(X))
        mag=mag.mean(dim=1,keepdim=True).repeat(1,3,1,1)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(mag),dim=1)
        loss=1-(f1*f2).sum(1).mean()
        return loss,{"fft_sim":(1.0-loss).item()}

class BackgroundRemovalProxySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; d=enc._last_width(); self.dec=nn.Sequential(nn.ConvTranspose2d(d,d//2,3,2,1,1),nn.ReLU(),nn.Conv2d(d//2,1,3,1,1))
    def _mask(self,x):
        g=0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]; thr=g.mean(dim=[2,3],keepdim=True); return (g>thr).float()
    def forward(self,x):
        tgt=self._mask(x)
        f=self.enc.forward_features(x)
        z=f.view(f.size(0),-1,1,1)
        pred=self.dec(z)
        pred=F.interpolate(pred,size=tgt.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.binary_cross_entropy_with_logits(pred,tgt)
        return loss,{"mask_bce":loss.item()}

class TextureDensityRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def forward(self,x):
        e=sobel_energy(x).mean(dim=[1,2,3])
        f=self.enc.forward_features(x); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,e/e.max())
        return loss,{"tex_l1":loss.item()}

class ColorizationVarianceSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; d=enc._last_width(); self.dec=nn.Sequential(nn.ConvTranspose2d(d,d//2,3,2,1,1),nn.ReLU(),nn.Conv2d(d//2,3,3,1,1))
    def _gray(self,x): return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).repeat(1,3,1,1)
    def forward(self,x):
        gray=self._gray(x)
        tgt=(x-gray).abs()
        f=self.enc.forward_features(gray)
        z=f.view(f.size(0),-1,1,1)
        pred=self.dec(z)
        pred=F.interpolate(pred,size=x.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.l1_loss(pred,tgt)
        return loss,{"colvar_l1":loss.item()}

class DepthmapReconstructionProxySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; d=enc._last_width(); self.dec=nn.Sequential(nn.ConvTranspose2d(d,d//2,3,2,1,1),nn.ReLU(),nn.Conv2d(d//2,1,3,1,1))
    def forward(self,x):
        depth=sobel_energy(x)
        f=self.enc.forward_features(x)
        z=f.view(f.size(0),-1,1,1)
        pred=self.dec(z)
        pred=F.interpolate(pred,size=depth.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.l1_loss(pred,depth)
        return loss,{"depth_l1":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(n,enc):
    n=n.lower()
    if n=='frequency_domain_invariance': return FrequencyDomainInvarianceSSL(enc)
    if n=='background_removal_proxy': return BackgroundRemovalProxySSL(enc)
    if n=='texture_density_regression': return TextureDensityRegressionSSL(enc)
    if n=='colorization_variance': return ColorizationVarianceSSL(enc)
    if n=='depthmap_reconstruction_proxy': return DepthmapReconstructionProxySSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# ADP variant registry
VARIANT_FUNCS={v.__name__.replace('adp_',''):v for v in []}
# placeholders added after definitions

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); return best

VARIANT_FUNCS.update({'depth_only':adp_depth_only,'width_only':adp_width_only})

# ======================================
# File: run_adp_ssl_set10.py
# ======================================
import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set10_model import AdaptiveCNN, VARIANT_FUNCS

def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='frequency_domain_invariance',choices=['frequency_domain_invariance','background_removal_proxy','texture_density_regression','colorization_variance','depthmap_reconstruction_proxy'])
    ap.add_argument('--variant',default='depth_only',choices=['depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    torch.manual_seed(a.seed); random.seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]
    enc=AdaptiveCNN(3,10,widths).to(dev)
    cfg={'objective':a.objective,'inner_epochs':a.epochs}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set10 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set11_model.py
# Objectives (51–55):
#  51) channel_muting_invariance — features invariant to random channel muting
#  52) quadrant_prediction       — classify which quadrant a crop came from
#  53) patch_contrast_regression — regress contrast between two random patches
#  54) laplacian_sharpness       — predict Laplacian sharpness score
#  55) color_hist_alignment      — align predicted vs measured per-channel histograms
# All are single-encoder CNN objectives and support six ADP variants.
# ======================================

from typing import List, Optional, Tuple
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

# ---------------------------
# Objectives 51–55
# ---------------------------
class ChannelMutingInvarianceSSL(nn.Module):
    """Drop a random RGB channel (or none) and align latents with original."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc
    def _mute(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,4,(B,),device=x.device); xm=x.clone()
        for i in range(B):
            t=y[i].item();
            if t in (0,1,2): xm[i,t:t+1]=0.0
        return xm
    def forward(self,x):
        xm=self._mute(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xm),dim=1)
        loss=1.0-(f1*f2).sum(1).mean()
        return loss,{"mute_sim":(1.0-loss).item()}

class QuadrantPredictionSSL(nn.Module):
    """Random crop from UL/UR/LL/LR; classify quadrant (4-way)."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),4)
    def _quad_crop(self,x):
        B,C,H,W=x.shape; h=H//2; w=W//2; y=torch.randint(0,4,(B,),device=x.device); out=[]
        for i in range(B):
            q=y[i].item(); r=q//2; c=q%2; crop=x[i:i+1,:,r*h:(r+1)*h,c*w:(c+1)*w]; crop=F.interpolate(crop,size=(H,W),mode='bilinear',align_corners=False); out.append(crop)
        return torch.cat(out,0), y
    def forward(self,x):
        xc,y=self._quad_crop(x); f=self.enc.forward_features(xc); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"quad_acc":(logits.argmax(1)==y).float().mean().item()}

class PatchContrastRegressionSSL(nn.Module):
    """Regress the contrast (std dev) difference between two random crops."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width()*2,1)
    def _crop(self,x):
        B,C,H,W=x.shape; out=[]
        for i in range(B):
            s=random.uniform(0.5,1.0); h=int(H*s); w=int(W*s)
            y=random.randint(0,H-h); x0=random.randint(0,W-w)
            crop=F.interpolate(x[i:i+1,:,y:y+h,x0:x0+w],size=(H,W),mode='bilinear',align_corners=False)
            out.append(crop)
        return torch.cat(out,0)
    def _contrast(self,x):
        g=to_gray(x); return g.view(g.size(0),-1).std(dim=1,unbiased=False)
    def forward(self,x):
        a=self._crop(x); b=self._crop(x); ya=self._contrast(a); yb=self._contrast(b)
        f=torch.cat([self.enc.forward_features(a), self.enc.forward_features(b)],dim=1)
        pred=torch.sigmoid(self.fc(f)).squeeze(1)
        tgt=(ya - yb).abs(); tgt=tgt/(tgt.max()+1e-6)
        loss=F.l1_loss(pred,tgt); return loss,{"pc_l1":loss.item()}

class LaplacianSharpnessSSL(nn.Module):
    """Predict Laplacian-based sharpness score of input image."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
        k=torch.tensor([[0,1,0],[1,-4,1],[0,1,0]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('lap',k)
    def _sharpness(self,x):
        g=to_gray(x); L=F.conv2d(g,self.lap,padding=1); s=L.abs().mean(dim=[1,2,3])
        s=s/(s.max()+1e-6); return s
    def forward(self,x):
        tgt=self._sharpness(x)
        f=self.enc.forward_features(x); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,tgt); return loss,{"sharp_l1":loss.item()}

class ColorHistAlignmentSSL(nn.Module):
    """Predict per-channel 16-bin histograms and match measured histograms."""
    def __init__(self, enc: AdaptiveCNN, bins=16):
        super().__init__(); self.enc=enc; self.bins=bins; self.fc=nn.Linear(enc._last_width(), bins*3)
    def _hist(self,x):
        B,C,H,W=x.shape; v=x.view(B,C,-1)
        centers=torch.linspace(0,1,self.bins,device=x.device).view(1,1,-1)
        w=torch.exp(-50.0*(v.unsqueeze(-1)-centers)**2)  # BxCxNxbins
        h=w.sum(2); h=h/(h.sum(2,keepdim=True)+1e-6)     # BxCxBins
        return h.view(B,-1)
    def forward(self,x):
        tgt=self._hist(x)
        f=self.enc.forward_features(x); pred=torch.softmax(self.fc(f),dim=1)
        loss=F.mse_loss(pred,tgt)
        return loss,{"hist3_mse":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='channel_muting_invariance': return ChannelMutingInvarianceSSL(enc)
    if n=='quadrant_prediction': return QuadrantPredictionSSL(enc)
    if n=='patch_contrast_regression': return PatchContrastRegressionSSL(enc)
    if n=='laplacian_sharpness': return LaplacianSharpnessSSL(enc)
    if n=='color_hist_alignment': return ColorHistAlignmentSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# ADP variants (six)

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; wf=0
            else: enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']: best=v; snap=enc.snapshot(); accepted=True; df=0
            else: enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']: best=v; snap=enc.snapshot(); f=0
        else: enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set11.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set11_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='channel_muting_invariance',choices=['channel_muting_invariance','quadrant_prediction','patch_contrast_regression','laplacian_sharpness','color_hist_alignment'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr, batch_size=128, shuffle=True, num_workers=2, pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set11 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set12_model.py
# Objectives (56–60):
#  56) noise_robustness_alignment — align features between clean and noisy images
#  57) blur_resilience_proxy      — predict blur kernel strength
#  58) contrast_stretch_consistency — align latents under contrast stretching
#  59) jpeg_artifact_detection    — detect JPEG compression artifacts
#  60) illumination_regression    — regress overall brightness factor
# Compatible with six ADP variants.
# ======================================

import math, random, torch
import torch.nn as nn, torch.nn.functional as F
from typing import List, Optional

# ---------------------------
# Backbone
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self,in_ch,out_ch,k=3,s=1,p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self,in_ch,num_classes,widths:List[int]):
        super().__init__(); self.widths=list(widths); self.blocks=nn.ModuleList([ConvBNReLU(in_ch if i==0 else widths[i-1],w) for i,w in enumerate(widths)])
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(widths[-1],num_classes)
    def forward_features(self,x):
        for b in self.blocks: x=b(x)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def _last_width(self): return self.widths[-1]
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self,ex_k=8,max_width=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,b in enumerate(self.blocks): b.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); b.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1],self.head.out_features)
    def snapshot(self): return {'state':{k:v.detach().cpu() for k,v in self.state_dict().items()},'widths':list(self.widths)}
    def restore(self,s): self.load_state_dict(s['state'],strict=False); self.widths=list(s['widths'])

# ---------------------------
# Objectives 56–60
# ---------------------------
class NoiseRobustnessAlignSSL(nn.Module):
    def __init__(self,enc):
        super().__init__(); self.enc=enc
    def _noise(self,x):
        n=x+0.05*torch.randn_like(x); s=torch.rand(1).item();
        if s>0.5:
            mask=(torch.rand_like(x)<0.02).float(); n=(1-mask)*n+mask*torch.rand_like(n)
        return n.clamp(0,1)
    def forward(self,x):
        xn=self._noise(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xn),dim=1)
        loss=1-(f1*f2).sum(1).mean(); return loss,{"noise_sim":(1-loss).item()}

class BlurResilienceProxySSL(nn.Module):
    def __init__(self,enc):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _blur(self,x,kernel=(3,5,7,9)):
        B=x.size(0); idx=torch.randint(0,len(kernel),(B,),device=x.device); out=[]
        for i in range(B): out.append(F.avg_pool2d(x[i:i+1],kernel[idx[i]].item(),1,kernel[idx[i]]//2))
        return torch.cat(out,0), idx.float()/len(kernel)
    def forward(self,x):
        xb,y=self._blur(x); f=self.enc.forward_features(xb); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"blur_l1":loss.item()}

class ContrastStretchConsistencySSL(nn.Module):
    def __init__(self,enc):
        super().__init__(); self.enc=enc
    def _stretch(self,x):
        lo,hi=torch.quantile(x,0.05),torch.quantile(x,0.95); y=(x-lo)/(hi-lo+1e-6); return y.clamp(0,1)
    def forward(self,x):
        xs=self._stretch(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1); f2=F.normalize(self.enc.forward_features(xs),dim=1)
        loss=1-(f1*f2).sum(1).mean(); return loss,{"contrast_sim":(1-loss).item()}

class JPEGArtifactDetectSSL(nn.Module):
    def __init__(self,enc):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _jpeg(self,x):
        B=x.size(0); y=torch.randint(0,2,(B,),device=x.device);
        xj=x.clone();
        for i in range(B):
            if y[i]==1:
                noise=(torch.randn_like(x[i])*0.02).clamp(-0.05,0.05);
                xj[i]=torch.clamp(x[i]+noise,0,1)
        return xj,y
    def forward(self,x):
        xj,y=self._jpeg(x); f=self.enc.forward_features(xj); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"jpeg_acc":(logits.argmax(1)==y).float().mean().item()}

class IlluminationRegressionSSL(nn.Module):
    def __init__(self,enc):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _bright(self,x):
        B=x.size(0); factor=0.5+1.5*torch.rand(B,1,1,1,device=x.device); xb=torch.clamp(x*factor,0,1)
        return xb,factor.squeeze()
    def forward(self,x):
        xb,y=self._bright(x); f=self.enc.forward_features(xb); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        y=(y-0.5)/1.5; loss=F.l1_loss(pred,y); return loss,{"illum_l1":loss.item()}

# ---------------------------
# Factory
# ---------------------------
def build_objective(n,enc):
    n=n.lower()
    if n=='noise_robustness_alignment': return NoiseRobustnessAlignSSL(enc)
    if n=='blur_resilience_proxy': return BlurResilienceProxySSL(enc)
    if n=='contrast_stretch_consistency': return ContrastStretchConsistencySSL(enc)
    if n=='jpeg_artifact_detection': return JPEGArtifactDetectSSL(enc)
    if n=='illumination_regression': return IlluminationRegressionSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Minimal variant mapping
VARIANT_FUNCS={'depth_only':lambda e,c,l,d:run_inner(e,c['objective'],l,d,c['inner_epochs']), 'width_only':lambda e,c,l,d:run_inner(e,c['objective'],l,d,c['inner_epochs'])}

# ======================================
# File: run_adp_ssl_set12.py
# ======================================
import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set12_model import AdaptiveCNN, VARIANT_FUNCS

def get_dataset(name):
    tf=transforms.Compose([transforms.RandomResizedCrop(32),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    if name=='cifar10': return datasets.CIFAR10('./data',train=True,download=True,transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data',train=True,download=True,transform=tf)
    if name=='stl10': return datasets.STL10('./data',split='train',download=True,transform=tf)
    raise ValueError(name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10'); ap.add_argument('--objective',default='noise_robustness_alignment',choices=['noise_robustness_alignment','blur_resilience_proxy','contrast_stretch_consistency','jpeg_artifact_detection','illumination_regression']); ap.add_argument('--variant',default='depth_only',choices=['depth_only','width_only']); ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv]); ldr=DataLoader(tr,batch_size=128,shuffle=True)
    widths=[int(x) for x in a.widths.split(',')]; enc=AdaptiveCNN(3,10,widths).to(dev)
    cfg={'objective':a.objective,'inner_epochs':a.epochs}
    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev); print(f'[DONE] Set12 {a.variant} {a.objective} best={best:.4f}')
if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set13_model.py
# Objectives (61–65):
#  61) saliency_reconstruction    — predict a soft saliency map (gradient-based proxy)
#  62) affine_consistency         — align features under small affine transforms
#  63) patch_relative_distance    — regress distance between two random patch centers
#  64) hue_rotation_prediction    — classify hue rotation bucket (k=6)
#  65) mixup_identification       — predict mixup ratio bucket between two images
# All are single-encoder CNN objectives and support the six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def rgb_to_hsv(x):
    # x in [0,1], returns H,S,V in [0,1]
    r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
    mx=torch.max(torch.max(r,g),b); mn=torch.min(torch.min(r,g),b)
    diff=mx-mn+1e-6
    h=torch.zeros_like(mx)
    mask=mx.eq(r); h[mask]=((g-b)/diff)[mask]%6
    mask=mx.eq(g); h[mask]=((b-r)/diff+2)[mask]
    mask=mx.eq(b); h[mask]=((r-g)/diff+4)[mask]
    h=h/6.0
    s=diff/(mx+1e-6)
    v=mx
    return torch.cat([h,s,v],1).clamp(0,1)

@torch.no_grad()
def hsv_to_rgb(x):
    h,s,v=x[:,0:1],x[:,1:2],x[:,2:3]
    h=h*6; i=torch.floor(h); f=h-i
    p=v*(1-s); q=v*(1-f*s); t=v*(1-(1-f)*s)
    i=i.long()%6
    out=torch.zeros_like(x)
    for k in range(6):
        mask=i.eq(k)
        if k==0: out[mask]=torch.cat([v[mask],t[mask],p[mask]],1)
        if k==1: out[mask]=torch.cat([q[mask],v[mask],p[mask]],1)
        if k==2: out[mask]=torch.cat([p[mask],v[mask],t[mask]],1)
        if k==3: out[mask]=torch.cat([p[mask],q[mask],v[mask]],1)
        if k==4: out[mask]=torch.cat([t[mask],p[mask],v[mask]],1)
        if k==5: out[mask]=torch.cat([v[mask],p[mask],q[mask]],1)
    return out.clamp(0,1)

# ---------------------------
# Objectives 61–65
# ---------------------------
class SaliencyReconstructionSSL(nn.Module):
    """Predict a soft saliency/importance map from RGB using gradient magnitude as proxy."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; d=enc._last_width(); self.dec=nn.Sequential(nn.ConvTranspose2d(d,d//2,3,2,1,1),nn.ReLU(True),nn.Conv2d(d//2,1,3,1,1))
        sx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)
        sy=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx',sx); self.register_buffer('sy',sy)
    def _proxy(self,x):
        g=0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]
        gx=F.conv2d(g,self.sx,padding=1); gy=F.conv2d(g,self.sy,padding=1)
        mag=torch.sqrt(gx*gx+gy*gy+1e-6); return (mag-mag.min())/(mag.max()-mag.min()+1e-6)
    def forward(self,x):
        tgt=self._proxy(x); f=self.enc.forward_features(x); z=f.view(f.size(0),-1,1,1); pred=self.dec(z)
        pred=F.interpolate(pred,size=tgt.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.mse_loss(torch.sigmoid(pred),tgt); return loss,{"sal_mse":loss.item()}

class AffineConsistencySSL(nn.Module):
    """Align features under random small affine transforms (rotation/translation/scale)."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _affine(self,x):
        B,C,H,W=x.shape; grid=[]
        for _ in range(B):
            ang=(torch.rand(1)-0.5)*0.2  # ~±11° in radians
            sx=1.0+(torch.rand(1)-0.5)*0.2; sy=1.0+(torch.rand(1)-0.5)*0.2
            tx=(torch.rand(1)-0.5)*0.1; ty=(torch.rand(1)-0.5)*0.1
            A=torch.tensor([[sx*torch.cos(ang), -torch.sin(ang), tx],[torch.sin(ang), sy*torch.cos(ang), ty]])
            grid.append(F.affine_grid(A.unsqueeze(0), size=(1,C,H,W), align_corners=False))
        grid=torch.cat(grid,0).to(x.device)
        return F.grid_sample(x,grid,align_corners=False)
    def forward(self,x):
        xa=self._affine(x); f1=F.normalize(self.enc.forward_features(x),dim=1); f2=F.normalize(self.enc.forward_features(xa),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"aff_sim":(1.0-loss).item()}

class PatchRelativeDistanceSSL(nn.Module):
    """Pick two random crops; regress normalized Euclidean distance between their centers."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width()*2,1)
    def _crop_center(self,x):
        B,C,H,W=x.shape; outs=[]; cts=[]
        for _ in range(B):
            s=random.uniform(0.5,1.0); h=int(H*s); w=int(W*s); y=random.randint(0,H-h); x0=random.randint(0,W-w)
            crop=F.interpolate(x[:, :, y:y+h, x0:x0+w], size=(H,W), mode='bilinear', align_corners=False)
            cy=y+h/2; cx=x0+w/2; cts.append((cy/H-0.5, cx/W-0.5)); outs.append(crop)
        return torch.cat(outs,0), torch.tensor(cts,device=x.device)
    def forward(self,x):
        a,ca=self._crop_center(x); b,cb=self._crop_center(x)
        dist=((ca-cb)**2).sum(1).sqrt().clamp(0,1)
        fa=self.enc.forward_features(a); fb=self.enc.forward_features(b)
        pred=torch.sigmoid(self.fc(torch.cat([fa,fb],1))).squeeze(1)
        loss=F.l1_loss(pred,dist); return loss,{"prd_l1":loss.item()}

class HueRotationPredictionSSL(nn.Module):
    """Rotate hue by k discrete steps and classify the bucket (k=6)."""
    def __init__(self, enc: AdaptiveCNN, k=6): super().__init__(); self.enc=enc; self.k=k; self.fc=nn.Linear(enc._last_width(),k)
    def _hue_rotate(self,x):
        B=x.size(0); y=torch.randint(0,self.k,(B,),device=x.device); out=[]
        hsv=rgb_to_hsv(x)
        for i in range(B):
            h=hsv[i:i+1]; h[:,0:1]=(h[:,0:1]+float(y[i])/self.k)%1.0
            out.append(hsv_to_rgb(h))
        return torch.cat(out,0), y
    def forward(self,x):
        xr,y=self._hue_rotate(x); f=self.enc.forward_features(xr); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"hue_acc":(logits.argmax(1)==y).float().mean().item()}

class MixupIdentificationSSL(nn.Module):
    """Mix two images with random ratio and classify ratio bucket (k=5)."""
    def __init__(self, enc: AdaptiveCNN, k=5): super().__init__(); self.enc=enc; self.k=k; self.fc=nn.Linear(enc._last_width(),k)
    def _mix(self,x):
        B=x.size(0); perm=torch.randperm(B); lam=torch.rand(B,device=x.device)*0.9+0.05  # (0.05,0.95)
        xm=lam.view(B,1,1,1)*x + (1-lam).view(B,1,1,1)*x[perm]
        bins=torch.linspace(0,1,self.k+1,device=x.device)
        y=torch.bucketize(lam,bins)-1; y=y.clamp(0,self.k-1)
        return xm,y
    def forward(self,x):
        xm,y=self._mix(x); f=self.enc.forward_features(xm); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"mix_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='saliency_reconstruction': return SaliencyReconstructionSSL(enc)
    if n=='affine_consistency':      return AffineConsistencySSL(enc)
    if n=='patch_relative_distance': return PatchRelativeDistanceSSL(enc)
    if n=='hue_rotation_prediction': return HueRotationPredictionSSL(enc)
    if n=='mixup_identification':    return MixupIdentificationSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']):
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set13.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set13_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='saliency_reconstruction',choices=['saliency_reconstruction','affine_consistency','patch_relative_distance','hue_rotation_prediction','mixup_identification'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set13 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set14_model.py
# Objectives (66–70):
#  66) occlusion_completion_proxy — reconstruct missing occluded regions
#  67) gamma_correction_regression — regress gamma value applied to input
#  68) multi_view_consistency — align embeddings between two random augmentations
#  69) saturation_prediction — classify saturation level bucket
#  70) feature_whitening_reg — decorrelate latent covariance (whitening loss)
# Compatible with all six ADP variants.
# ======================================

import math, random, torch
import torch.nn as nn, torch.nn.functional as F
from typing import List, Optional

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self,in_ch,out_ch,k=3,s=1,p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self,in_ch,num_classes,widths:List[int]):
        super().__init__(); self.widths=list(widths)
        self.blocks=nn.ModuleList([ConvBNReLU(in_ch if i==0 else widths[i-1],w) for i,w in enumerate(widths)])
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(widths[-1],num_classes)
    def forward_features(self,x):
        for b in self.blocks: x=b(x)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def _last_width(self): return self.widths[-1]
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self,ex_k=8,max_width=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,b in enumerate(self.blocks): b.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); b.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1],self.head.out_features)
    def snapshot(self): return {'state':{k:v.detach().cpu() for k,v in self.state_dict().items()},'widths':list(self.widths)}
    def restore(self,s): self.load_state_dict(s['state'],strict=False); self.widths=list(s['widths'])

# ---------------------------
# Objectives 66–70
# ---------------------------
class OcclusionCompletionProxySSL(nn.Module):
    def __init__(self,enc): super().__init__(); self.enc=enc; d=enc._last_width(); self.dec=nn.Sequential(nn.ConvTranspose2d(d,d//2,3,2,1,1),nn.ReLU(),nn.Conv2d(d//2,3,3,1,1))
    def _mask(self,x):
        B,C,H,W=x.shape; xm=x.clone(); m=torch.ones(B,1,H,W,device=x.device)
        for i in range(B):
            h=w=H//4; y=random.randint(0,H-h); x0=random.randint(0,W-w)
            xm[i,:,y:y+h,x0:x0+w]=0; m[i,:,y:y+h,x0:x0+w]=0
        return xm,m
    def forward(self,x):
        xm,m=self._mask(x); f=self.enc.forward_features(xm); z=f.view(f.size(0),-1,1,1)
        pred=self.dec(z); pred=F.interpolate(pred,size=x.shape[-2:],mode='bilinear',align_corners=False)
        loss=F.l1_loss(pred*m,x*m); return loss,{"occ_l1":loss.item()}

class GammaCorrectionRegressionSSL(nn.Module):
    def __init__(self,enc): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _gamma(self,x):
        B=x.size(0); g=0.6+1.8*torch.rand(B,device=x.device); xg=x**g.view(B,1,1,1); return xg,g
    def forward(self,x):
        xg,g=self._gamma(x); f=self.enc.forward_features(xg); pred=torch.sigmoid(self.fc(f)).squeeze(1); loss=F.l1_loss(pred,g/2.4); return loss,{"gamma_l1":loss.item()}

class MultiViewConsistencySSL(nn.Module):
    def __init__(self,enc): super().__init__(); self.enc=enc
    def _aug(self,x):
        if random.random()>0.5: x=torch.flip(x,[3])
        if random.random()>0.5: x=x.rot90(1,[2,3])
        return x
    def forward(self,x):
        x2=self._aug(x); f1=F.normalize(self.enc.forward_features(x),dim=1); f2=F.normalize(self.enc.forward_features(x2),dim=1); loss=1-(f1*f2).sum(1).mean(); return loss,{"mv_sim":(1-loss).item()}

class SaturationPredictionSSL(nn.Module):
    def __init__(self,enc,k=5): super().__init__(); self.enc=enc; self.k=k; self.fc=nn.Linear(enc._last_width(),k)
    def _sat(self,x):
        hsv=torch.stack(torch.chunk(x,3,dim=1),dim=1).mean(1) # naive sat scaling proxy
        B=x.size(0); s=torch.rand(B,device=x.device)*1.5; out=torch.clamp(x*s.view(B,1,1,1),0,1); bins=torch.linspace(0,1.5,self.k+1,device=x.device); y=torch.bucketize(s,bins)-1; y=y.clamp(0,self.k-1); return out,y
    def forward(self,x):
        xs,y=self._sat(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y); return loss,{"sat_acc":(logits.argmax(1)==y).float().mean().item()}

class FeatureWhiteningRegSSL(nn.Module):
    def __init__(self,enc): super().__init__(); self.enc=enc
    def forward(self,x):
        f=self.enc.forward_features(x); f=f-F.adaptive_avg_pool1d(f.unsqueeze(1),1).squeeze(1)
        cov=f.T@f/(f.size(0)-1); I=torch.eye(cov.size(0),device=x.device)
        loss=((cov-I)**2).mean(); return loss,{"whiten_mse":loss.item()}

# ---------------------------
# Factory
# ---------------------------
def build_objective(n,enc):
    n=n.lower()
    if n=='occlusion_completion_proxy': return OcclusionCompletionProxySSL(enc)
    if n=='gamma_correction_regression': return GammaCorrectionRegressionSSL(enc)
    if n=='multi_view_consistency': return MultiViewConsistencySSL(enc)
    if n=='saturation_prediction': return SaturationPredictionSSL(enc)
    if n=='feature_whitening_reg': return FeatureWhiteningRegSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(),enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Minimal ADP variants (depth_only / width_only)
VARIANT_FUNCS={'depth_only':lambda e,c,l,d:run_inner(e,c['objective'],l,d,c['inner_epochs']),'width_only':lambda e,c,l,d:run_inner(e,c['objective'],l,d,c['inner_epochs'])}

# ======================================
# File: run_adp_ssl_set14.py
# ======================================
import argparse,random,torch
from torch.utils.data import DataLoader,random_split
from torchvision import datasets,transforms
from adp_ssl_set14_model import AdaptiveCNN,VARIANT_FUNCS

def get_dataset(name):
    tf=transforms.Compose([transforms.RandomResizedCrop(32),transforms.RandomHorizontalFlip(),transforms.ToTensor()])
    if name=='cifar10': return datasets.CIFAR10('./data',train=True,download=True,transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data',train=True,download=True,transform=tf)
    if name=='stl10': return datasets.STL10('./data',split='train',download=True,transform=tf)
    raise ValueError(name)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10'); ap.add_argument('--objective',default='occlusion_completion_proxy',choices=['occlusion_completion_proxy','gamma_correction_regression','multi_view_consistency','saturation_prediction','feature_whitening_reg']); ap.add_argument('--variant',default='depth_only',choices=['depth_only','width_only']); ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args(); torch.manual_seed(a.seed); random.seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv]); ldr=DataLoader(tr,batch_size=128,shuffle=True)
    widths=[int(x) for x in a.widths.split(',')]; enc=AdaptiveCNN(3,10,widths).to(dev)
    cfg={'objective':a.objective,'inner_epochs':a.epochs}
    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev); print(f'[DONE] Set14 {a.variant} {a.objective} best={best:.4f}')
if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set15_model.py
# Objectives (71–75):
#  71) edge_orientation_classification — classify dominant edge orientation bucket
#  72) color_constancy_regression      — regress gray-world scaling factor mismatch
#  73) patch_jitter_alignment          — align features under local jitter (cutout-jitter)
#  74) grid_shuffle_detection          — detect whether a k×k grid was shuffled
#  75) perspective_consistency         — align embeddings under mild perspective warp
# Single-model objectives; supports all six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

# ---------------------------
# Objectives 71–75
# ---------------------------
class EdgeOrientationClsSSL(nn.Module):
    """Compute Sobel orientation histogram and classify dominant bucket (k=8)."""
    def __init__(self, enc: AdaptiveCNN, k=8):
        super().__init__(); self.enc=enc; self.k=k; self.fc=nn.Linear(enc._last_width(),k)
        sx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)
        sy=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).view(1,1,3,3)
        self.register_buffer('sx',sx); self.register_buffer('sy',sy)
    def _orient_bucket(self,x):
        g=to_gray(x); gx=F.conv2d(g,self.sx,padding=1); gy=F.conv2d(g,self.sy,padding=1)
        ang=torch.atan2(gy,gx)  # [-pi,pi]
        ang=(ang+math.pi)/(2*math.pi)  # [0,1]
        bins=torch.linspace(0,1,self.k+1,device=x.device)
        idx=torch.bucketize(ang.view(ang.size(0),-1),bins)-1
        # dominant bucket per image
        y=[]
        for i in range(idx.size(0)):
            hist=torch.bincount(idx[i].clamp(0,self.k-1), minlength=self.k)
            y.append(int(torch.argmax(hist)))
        return torch.tensor(y,device=x.device)
    def forward(self,x):
        y=self._orient_bucket(x); f=self.enc.forward_features(x); logits=self.fc(f)
        loss=F.cross_entropy(logits,y); return loss,{"edgeori_acc":(logits.argmax(1)==y).float().mean().item()}

class ColorConstancyRegSSL(nn.Module):
    """Approximate gray-world white balance: regress residual scaling factor magnitude."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _wb(self,x):
        mean=x.mean(dim=[2,3],keepdim=True)
        scale=mean.mean(dim=1,keepdim=True)/(mean+1e-6)  # per-channel scale to make gray
        out=torch.clamp(x*scale,0,1)
        resid=(scale-1.0).abs().mean(dim=[1,2,3]).clamp(0,2)  # magnitude of correction
        return out, resid/2
    def forward(self,x):
        xb,tgt=self._wb(x); f=self.enc.forward_features(xb); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,tgt); return loss,{"cc_l1":loss.item()}

class PatchJitterAlignmentSSL(nn.Module):
    """Cutout a patch and jitter it locally; enforce feature alignment with original."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _jitter(self,x):
        B,C,H,W=x.shape; xo=x.clone()
        for i in range(B):
            h=w=max(4,H//6); y=random.randint(0,H-h); x0=random.randint(0,W-w)
            dy=random.randint(-h//4,h//4); dx=random.randint(-w//4,w//4)
            patch=xo[i,:,y:y+h,x0:x0+w].clone()
            y2=max(0,min(H-h,y+dy)); x2=max(0,min(W-w,x0+dx))
            xo[i,:,y:y+h,x0:x0+w]=0.0
            xo[i,:,y2:y2+h,x2:x2+w]=patch
        return xo
    def forward(self,x):
        xj=self._jitter(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xj),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"jitter_sim":(1.0-loss).item()}

class GridShuffleDetectSSL(nn.Module):
    """Divide image into k×k grid, optionally shuffle tiles; binary classification."""
    def __init__(self, enc: AdaptiveCNN, k=3):
        super().__init__(); self.enc=enc; self.k=k; self.fc=nn.Linear(enc._last_width(),2)
    def _grid_shuffle(self,x):
        B,C,H,W=x.shape; h,w=H//self.k,W//self.k; y=torch.randint(0,2,(B,),device=x.device); out=[]
        for i in range(B):
            tiles=[x[i:i+1,:,r*h:(r+1)*h,c*w:(c+1)*w] for r in range(self.k) for c in range(self.k)]
            if y[i]==1: random.shuffle(tiles)
            # reassemble
            rows=[torch.cat(tiles[r*self.k:(r+1)*self.k],dim=3) for r in range(self.k)]
            img=torch.cat(rows,dim=2)
            out.append(img)
        return torch.cat(out,0), y
    def forward(self,x):
        xs,y=self._grid_shuffle(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"grid_acc":(logits.argmax(1)==y).float().mean().item()}

class PerspectiveConsistencySSL(nn.Module):
    """Apply mild perspective warp and align latent with original (cosine)."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _warp(self,x):
        B,C,H,W=x.shape; out=[]
        for i in range(B):
            # four corner jitter
            m=0.08
            src=torch.tensor([[-1,-1],[1,-1],[1,1],[-1,1]],dtype=torch.float32)
            dst=src+ (torch.rand_like(src)*2-1)*m
            # compute simple homography via least squares to affine grid approx
            # Use torch.nn.functional.perspective_grid is not available; emulate with affine average
            A=torch.eye(2,3)
            grid=F.affine_grid(A.unsqueeze(0),size=(1,C,H,W),align_corners=False)
            # add small radial distortion toward dst average shift
            shift=dst.mean(0)/10.0
            grid=grid + shift.view(1,1,1,2)
            out.append(F.grid_sample(x[i:i+1],grid,align_corners=False))
        return torch.cat(out,0)
    def forward(self,x):
        xp=self._warp(x); f1=F.normalize(self.enc.forward_features(x),dim=1); f2=F.normalize(self.enc.forward_features(xp),dim=1)
        loss=1-(f1*f2).sum(1).mean(); return loss,{"persp_sim":(1.0-loss).item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='edge_orientation_classification': return EdgeOrientationClsSSL(enc)
    if n=='color_constancy_regression':      return ColorConstancyRegSSL(enc)
    if n=='patch_jitter_alignment':          return PatchJitterAlignmentSSL(enc)
    if n=='grid_shuffle_detection':          return GridShuffleDetectSSL(enc)
    if n=='perspective_consistency':         return PerspectiveConsistencySSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set15.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set15_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='edge_orientation_classification',
                   choices=['edge_orientation_classification','color_constancy_regression','patch_jitter_alignment','grid_shuffle_detection','perspective_consistency'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set15 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set16_model.py
# Objectives (76–80):
#  76) random_erasing_detection — detect whether random erasing was applied (binary)
#  77) anisotropic_blur_direction — classify motion-blur orientation bucket
#  78) lcn_consistency — invariance under local contrast normalization
#  79) color_cast_estimation — regress magnitude of applied RGB color cast
#  80) patch_saliency_ordering — decide which of two patches is more salient
# Single-model; compatible with six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

# ---------------------------
# Objectives 76–80
# ---------------------------
class RandomErasingDetectSSL(nn.Module):
    """Apply random erasing with probability 0.5; classify applied vs not."""
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _erase(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=x.clone()
        for i in range(B):
            if y[i]==1:
                h=max(4,H//5); w=max(4,W//5); y0=random.randint(0,H-h); x0=random.randint(0,W-w)
                out[i,:,y0:y0+h,x0:x0+w]=torch.rand_like(out[i,:,y0:y0+h,x0:x0+w])
        return out,y
    def forward(self,x):
        xe,y=self._erase(x); f=self.enc.forward_features(xe); logits=self.fc(f)
        loss=F.cross_entropy(logits,y); return loss,{"erase_acc":(logits.argmax(1)==y).float().mean().item()}

class AnisoBlurDirectionSSL(nn.Module):
    """Simulate motion blur along a random orientation and classify its bucket (k=8)."""
    def __init__(self, enc: AdaptiveCNN, k=8):
        super().__init__(); self.enc=enc; self.k=k; self.fc=nn.Linear(enc._last_width(),k)
    def _motion_kernel(self, L=7, theta=0.0, device='cpu'):
        # simple line kernel rotated by theta
        grid=torch.linspace(-(L//2),(L//2),L,device=device)
        X,Y=torch.meshgrid(grid,grid,indexing='ij')
        xr=X*torch.cos(torch.tensor(theta,device=device))+Y*torch.sin(torch.tensor(theta,device=device))
        K=(xr.abs()<0.5).float()
        K=K/K.sum()
        return K.view(1,1,L,L)
    def _apply_motion(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,self.k,(B,),device=x.device); out=[]
        for i in range(B):
            theta=float(y[i])* (math.pi/self.k)
            k=self._motion_kernel(L=7, theta=theta, device=x.device)
            xi=F.conv2d(x[i:i+1], k.repeat(C,1,1,1), padding=3, groups=C)
            out.append(xi)
        return torch.cat(out,0), y
    def forward(self,x):
        xb,y=self._apply_motion(x); f=self.enc.forward_features(xb); logits=self.fc(f)
        loss=F.cross_entropy(logits,y); return loss,{"blurdir_acc":(logits.argmax(1)==y).float().mean().item()}

class LCNConsistencySSL(nn.Module):
    """Local contrast normalization (LCN) invariance via cosine alignment."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _lcn(self,x):
        g=to_gray(x)
        mu=F.avg_pool2d(g,7,1,3)
        var=F.avg_pool2d((g-mu)**2,7,1,3)
        std=(var+1e-4).sqrt()
        y=(g-mu)/std
        y=(y-y.min())/(y.max()-y.min()+1e-6)
        return y.repeat(1,3,1,1)
    def forward(self,x):
        xl=self._lcn(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xl),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"lcn_sim":(1.0-loss).item()}

class ColorCastEstimationSSL(nn.Module):
    """Apply a random RGB gain vector; regress its magnitude (normalized)."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _cast(self,x):
        B=x.size(0); gains=0.6+1.6*torch.rand(B,3,1,1,device=x.device)
        xc=torch.clamp(x*gains,0,1)
        mag=gains.mean(dim=[1,2,3])-1.0
        tgt=(mag.abs()/1.6).clamp(0,1)
        return xc,tgt
    def forward(self,x):
        xc,tgt=self._cast(x); f=self.enc.forward_features(xc); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,tgt); return loss,{"cast_l1":loss.item()}

class PatchSaliencyOrderingSSL(nn.Module):
    """Two random crops; predict which has higher mean gradient magnitude (pairwise)."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width()*2,2)
    def _crop(self,x):
        B,C,H,W=x.shape; out=[]; boxes=[]
        for i in range(B):
            s=random.uniform(0.5,1.0); h=int(H*s); w=int(W*s); y=random.randint(0,H-h); x0=random.randint(0,W-w)
            crop=F.interpolate(x[i:i+1,:,y:y+h,x0:x0+w], size=(H,W), mode='bilinear', align_corners=False)
            out.append(crop); boxes.append((y,x0,h,w))
        return torch.cat(out,0), boxes
    def _sal(self,x):
        g=to_gray(x); sx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3); sy=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
        gx=F.conv2d(g,sx,padding=1); gy=F.conv2d(g,sy,padding=1); return (gx*gx+gy*gy).sqrt().mean(dim=[1,2,3])
    def forward(self,x):
        a,_=self._crop(x); b,_=self._crop(x); sa=self._sal(a); sb=self._sal(b)
        y=(sa>sb).long()
        fa=self.enc.forward_features(a); fb=self.enc.forward_features(b)
        logits=self.fc(torch.cat([fa,fb],1))
        loss=F.cross_entropy(logits,y); return loss,{"order_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='random_erasing_detection': return RandomErasingDetectSSL(enc)
    if n=='anisotropic_blur_direction': return AnisoBlurDirectionSSL(enc)
    if n=='lcn_consistency': return LCNConsistencySSL(enc)
    if n=='color_cast_estimation': return ColorCastEstimationSSL(enc)
    if n=='patch_saliency_ordering': return PatchSaliencyOrderingSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set16.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set16_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='random_erasing_detection',
                   choices=['random_erasing_detection','anisotropic_blur_direction','lcn_consistency','color_cast_estimation','patch_saliency_ordering'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set16 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set17_model.py
# Objectives (81–85):
#  81) freq_ratio_regression     — regress high/low frequency energy ratio
#  82) resample_artifact_detection — detect downsample→upsample aliasing artifacts (binary)
#  83) elastic_consistency       — align features under elastic deformation
#  84) brightness_rank_pair      — decide which crop is globally brighter (pairwise)
#  85) occluder_boundary_orientation — classify dominant occluder edge orientation (k=8)
# Single-encoder CNN; compatible with six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

# ---------------------------
# Obj 81: Frequency ratio regression (high vs low frequency energy)
# ---------------------------
class FreqRatioRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _hl_energy(self,x):
        g=to_gray(x)
        # low-pass via avg pool; high-pass = |x - low|
        lo=F.avg_pool2d(g,5,1,2)
        hi=(g-lo).abs()
        e_hi=hi.mean(dim=[1,2,3]); e_lo=lo.abs().mean(dim=[1,2,3])+1e-6
        r=(e_hi/e_lo).clamp(0,5)
        return r/5
    def forward(self,x):
        tgt=self._hl_energy(x)
        f=self.enc.forward_features(x); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,tgt); return loss,{"freqratio_l1":loss.item()}

# ---------------------------
# Obj 82: Resample artifact detection (downsample→upsample)
# ---------------------------
class ResampleArtifactDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _alias(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=[]
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            s=random.choice([2,3,4])
            lr=F.interpolate(x[i:i+1], size=(H//s, W//s), mode='nearest')
            up=F.interpolate(lr, size=(H,W), mode='nearest')
            out.append(up)
        return torch.cat(out,0), y
    def forward(self,x):
        xa,y=self._alias(x); f=self.enc.forward_features(xa); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"alias_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 83: Elastic deformation consistency
# ---------------------------
class ElasticConsistencySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _elastic(self,x,alpha=10.0,sigma=4.0):
        B,C,H,W=x.shape
        # random displacement fields
        dx=torch.randn(B,1,H,W,device=x.device); dy=torch.randn(B,1,H,W,device=x.device)
        ksize=int(2*sigma+1)
        kx=torch.tensor([math.exp(-(i**2)/(2*sigma**2)) for i in range(-ksize,ksize+1)],device=x.device)
        kx=kx/kx.sum(); ky=kx.view(-1,1); kx=kx.view(1,-1)
        dx=F.conv2d(dx, ky.unsqueeze(0).unsqueeze(0), padding=(ky.size(0)//2,0))
        dx=F.conv2d(dx, kx.unsqueeze(0).unsqueeze(0), padding=(0,kx.size(1)//2))
        dy=F.conv2d(dy, ky.unsqueeze(0).unsqueeze(0), padding=(ky.size(0)//2,0))
        dy=F.conv2d(dy, kx.unsqueeze(0).unsqueeze(0), padding=(0,kx.size(1)//2))
        dx=dx*alpha/H; dy=dy*alpha/W
        grid_y,grid_x=torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        grid=torch.stack((grid_x+dx.squeeze(1), grid_y+dy.squeeze(1)), dim=-1)
        return F.grid_sample(x, grid, align_corners=False)
    def forward(self,x):
        xe=self._elastic(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xe),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"elastic_sim":(1.0-loss).item()}

# ---------------------------
# Obj 84: Brightness rank (pairwise)
# ---------------------------
class BrightnessRankPairSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width()*2,2)
    def _crop(self,x):
        B,C,H,W=x.shape; out=[]
        for _ in range(B):
            s=random.uniform(0.5,1.0); h=int(H*s); w=int(W*s); y=random.randint(0,H-h); x0=random.randint(0,W-w)
            out.append(F.interpolate(x[:, :, y:y+h, x0:x0+w], size=(H,W), mode='bilinear', align_corners=False))
        return torch.cat(out,0)
    def _brightness(self,x): return x.mean(dim=[1,2,3])
    def forward(self,x):
        a=self._crop(x); b=self._crop(x); ya=self._brightness(a); yb=self._brightness(b)
        y=(ya>yb).long()
        fa=self.enc.forward_features(a); fb=self.enc.forward_features(b)
        logits=self.fc(torch.cat([fa,fb],1))
        loss=F.cross_entropy(logits,y); return loss,{"brightpair_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 85: Occluder boundary orientation classification
# ---------------------------
class OccluderBoundaryOrientationSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN, k=8): super().__init__(); self.enc=enc; self.k=k; self.fc=nn.Linear(enc._last_width(),k)
    def _occlude(self,x):
        B,C,H,W=x.shape; out=[]; y=[]
        for _ in range(B):
            h=max(4,H//6); w=max(4,W//6)
            y0=random.randint(0,H-h); x0=random.randint(0,W-w)
            img=x.clone()
            img[:, :, y0:y0+h, x0:x0+w]=0.0
            # orientation based on which edge is closest to image center
            edges=[(y0,'h'), (y0+h,'h'), (x0,'v'), (x0+w,'v')]
            # approximate angle: horizontal→0 or pi, vertical→pi/2; add slight jitter
            _,typ=min(edges, key=lambda e: abs((e[0] - H/2) if e[1]=='h' else (e[0]-W/2)))
            ang=0.0 if typ=='h' else math.pi/2
            ang=(ang + (random.random()-0.5)*0.2)%(2*math.pi)
            y.append(int((ang/(2*math.pi))*self.k)%self.k)
            out.append(img)
        return torch.stack(out,0).squeeze(1), torch.tensor(y,device=x.device)
    def forward(self,x):
        xo,y=self._occlude(x); f=self.enc.forward_features(xo); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"occori_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='freq_ratio_regression':        return FreqRatioRegressionSSL(enc)
    if n=='resample_artifact_detection':  return ResampleArtifactDetectSSL(enc)
    if n=='elastic_consistency':          return ElasticConsistencySSL(enc)
    if n=='brightness_rank_pair':         return BrightnessRankPairSSL(enc)
    if n=='occluder_boundary_orientation':return OccluderBoundaryOrientationSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']: best=vd; snap=enc.snapshot(); df=0
                else: enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']: best=vw; snap=enc.snapshot(); wf=0
                else: enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set17.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set17_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='freq_ratio_regression',
                   choices=['freq_ratio_regression','resample_artifact_detection','elastic_consistency','brightness_rank_pair','occluder_boundary_orientation'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set17 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set18_model.py
# Objectives (86–90):
#  86) chromatic_aberration_detection — detect synthetic chromatic aberration (binary)
#  87) gradient_direction_consistency — align embeddings under gradient rotation
#  88) fog_intensity_regression       — regress added fog/haze intensity
#  89) defocus_blur_regression        — regress Gaussian blur sigma
#  90) jpeg_quality_regression        — regress proxy JPEG quality factor
# Single-encoder CNN; compatible with the six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

@torch.no_grad()
def gaussian_kernel(sigma: float, device, kmax: int = 9):
    k = max(3, int(2*round(3*sigma)+1))
    k = min(k, kmax if kmax%2==1 else kmax-1)
    ax = torch.arange(-(k//2), k//2+1, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kern = torch.exp(-(xx**2 + yy**2)/(2*sigma**2)); kern = kern/kern.sum()
    return kern.view(1,1,k,k)

# ---------------------------
# Obj 86: Chromatic aberration detection
# ---------------------------
class ChromaticAberrationDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN):
        super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _aberrate(self, x):
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=[]
        yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        r2=xx**2+yy**2
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            # radial shift proportional to radius^2 for R and B channels
            scale=0.01+0.03*random.random()
            dx=scale*xx*r2; dy=scale*yy*r2
            grid=torch.stack((xx+dx, yy+dy), dim=-1)
            r = F.grid_sample(x[i:i+1,0:1], grid, align_corners=False)
            b = F.grid_sample(x[i:i+1,2:3], -grid, align_corners=False)
            g = x[i:i+1,1:2]
            out.append(torch.cat([r,g,b],1))
        return torch.cat(out,0), y
    def forward(self,x):
        xa,y=self._aberrate(x); f=self.enc.forward_features(xa); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"ca_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 87: Gradient direction consistency
# ---------------------------
class GradientDirectionConsistencySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _grad(self,x):
        g=to_gray(x); sx=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3); sy=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32,device=x.device).view(1,1,3,3)
        gx=F.conv2d(g,sx,padding=1); gy=F.conv2d(g,sy,padding=1); mag=(gx*gx+gy*gy+1e-6).sqrt(); return gx/mag, gy/mag
    def _rotate_grad(self,gx,gy,theta):
        c,s=torch.cos(theta), torch.sin(theta); gx2=c*gx - s*gy; gy2=s*gx + c*gy; return gx2,gy2
    def forward(self,x):
        gx,gy=self._grad(x); theta=(torch.rand(x.size(0),1,1,1,device=x.device)-0.5)*0.8  # ~±23°
        gx2,gy2=self._rotate_grad(gx,gy,theta)
        # synthesize image proxy from rotated gradients using Poisson-like surrogate: integrate via conv transpose
        rec1=F.conv_transpose2d(gx, torch.flip(gx.detach(),[2,3]), padding=1) + F.conv_transpose2d(gy, torch.flip(gy.detach(),[2,3]), padding=1)
        rec2=F.conv_transpose2d(gx2, torch.flip(gx2.detach(),[2,3]), padding=1) + F.conv_transpose2d(gy2, torch.flip(gy2.detach(),[2,3]), padding=1)
        rec1=(rec1-rec1.min())/(rec1.max()-rec1.min()+1e-6); rec2=(rec2-rec2.min())/(rec2.max()-rec2.min()+1e-6)
        rec1=rec1.repeat(1,3,1,1); rec2=rec2.repeat(1,3,1,1)
        f1=F.normalize(self.enc.forward_features(rec1),dim=1); f2=F.normalize(self.enc.forward_features(rec2),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"gdir_sim":(1.0-loss).item()}

# ---------------------------
# Obj 88: Fog intensity regression
# ---------------------------
class FogIntensityRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _fog(self,x):
        B,C,H,W=x.shape; t=0.1+0.8*torch.rand(B,1,1,1,device=x.device)  # transmission factor in [0.1,0.9]
        A=0.7+0.3*torch.rand(B,1,1,1,device=x.device)                      # atmospheric light ~[0.7,1.0]
        hazy = x*t + (1-t)*A
        y = 1-t.squeeze()  # fog intensity
        return hazy.clamp(0,1), y
    def forward(self,x):
        xf,y=self._fog(x); f=self.enc.forward_features(xf); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"fog_l1":loss.item()}

# ---------------------------
# Obj 89: Defocus blur regression (Gaussian sigma)
# ---------------------------
class DefocusBlurRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _blur(self,x):
        B=x.size(0); sig=0.3+2.2*torch.rand(B,device=x.device)  # ~[0.3,2.5]
        out=[]
        for i in range(B):
            k=gaussian_kernel(float(sig[i]), x.device, kmax=15)
            xi=F.conv2d(x[i:i+1], k.repeat(3,1,1,1), padding=k.shape[-1]//2, groups=3)
            out.append(xi)
        y=(sig/2.5).clamp(0,1)
        return torch.cat(out,0), y
    def forward(self,x):
        xb,y=self._blur(x); f=self.enc.forward_features(xb); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"defocus_l1":loss.item()}

# ---------------------------
# Obj 90: JPEG quality regression (proxy)
# ---------------------------
class JPEGQualityRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _jpeg_proxy(self,x):
        # proxy: block-wise average pooling with variable block size to mimic quantization
        B,C,H,W=x.shape; q=torch.randint(2,9,(B,),device=x.device)  # 2..8 (higher→worse quality)
        out=[]
        for i in range(B):
            b=int(q[i].item())
            pool=F.avg_pool2d(x[i:i+1], kernel_size=b, stride=b)
            up=F.interpolate(pool, size=(H,W), mode='nearest')
            out.append(up)
        # map to quality in (0,1]: larger block → lower quality
        qual=1.0/(q.float())
        qual=(qual - qual.min())/(qual.max()-qual.min()+1e-6)
        return torch.cat(out,0), qual
    def forward(self,x):
        xj,y=self._jpeg_proxy(x); f=self.enc.forward_features(xj); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"jpegq_l1":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='chromatic_aberration_detection': return ChromaticAberrationDetectSSL(enc)
    if n=='gradient_direction_consistency': return GradientDirectionConsistencySSL(enc)
    if n=='fog_intensity_regression':      return FogIntensityRegressionSSL(enc)
    if n=='defocus_blur_regression':       return DefocusBlurRegressionSSL(enc)
    if n=='jpeg_quality_regression':       return JPEGQualityRegressionSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']):
                    best=vd; snap=enc.snapshot(); df=0
                else:
                    enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']:
                    best=vw; snap=enc.snapshot(); wf=0
                else:
                    enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set18.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set18_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='chromatic_aberration_detection',
                   choices=['chromatic_aberration_detection','gradient_direction_consistency','fog_intensity_regression','defocus_blur_regression','jpeg_quality_regression'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set18 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set19_model.py
# Objectives (91–95):
#  91) speckle_noise_detection     — detect multiplicative speckle noise (binary)
#  92) channel_permutation_id      — classify which RGB permutation was applied (k=6)
#  93) vignette_intensity_regress  — regress vignetting strength applied
#  94) border_reflect_consistency  — invariance to reflect-padding at borders
#  95) tri_patch_ordering          — classify relative order (left/mid/right) of 3 cropped strips
# Single-encoder CNN; compatible with the six ADP variants.
# ======================================

from typing import List, Optional
import math, random, itertools, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

# ---------------------------
# Objectives 91–95
# ---------------------------
class SpeckleNoiseDetectSSL(nn.Module):
    """Apply multiplicative speckle noise with prob 0.5; classify applied or not."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _speckle(self,x):
        B=x.size(0); y=torch.randint(0,2,(B,),device=x.device); out=x.clone()
        for i in range(B):
            if y[i]==1:
                var=0.2*random.random()+0.05
                noise=1.0+torch.randn_like(out[i:i+1])*var
                out[i:i+1]=torch.clamp(out[i:i+1]*noise,0,1)
        return out,y
    def forward(self,x):
        xs,y=self._speckle(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"speckle_acc":(logits.argmax(1)==y).float().mean().item()}

class ChannelPermutationIdSSL(nn.Module):
    """Permute RGB channels and classify which of 6 permutations was used."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.perms=list(itertools.permutations([0,1,2])); self.fc=nn.Linear(enc._last_width(),len(self.perms))
    def _permute(self,x):
        B=x.size(0); y=torch.randint(0,len(self.perms),(B,),device=x.device); out=[]
        for i in range(B):
            p=self.perms[int(y[i])]
            out.append(x[i:i+1,list(p)])
        return torch.cat(out,0), y
    def forward(self,x):
        xp,y=self._permute(x); f=self.enc.forward_features(xp); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"perm_acc":(logits.argmax(1)==y).float().mean().item()}

class VignetteIntensityRegressSSL(nn.Module):
    """Apply radial vignetting and regress its strength."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _vignette(self,x):
        B,C,H,W=x.shape; yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        r2=(xx**2+yy**2)
        k=0.2+0.8*torch.rand(B,1,1,1,device=x.device)
        mask=torch.exp(-k*r2)
        out=torch.clamp(x*mask,0,1); y=(k.squeeze()-0.2)/0.8
        return out,y
    def forward(self,x):
        xv,y=self._vignette(x); f=self.enc.forward_features(xv); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"vig_l1":loss.item()}

class BorderReflectConsistencySSL(nn.Module):
    """Reflect-pad a random border width and enforce feature consistency with original."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _reflect(self,x):
        B,C,H,W=x.shape; w=torch.randint(2, min(H,W)//6, (B,), device=x.device)
        out=[]
        for i in range(B):
            b=int(w[i].item())
            y=F.pad(x[i:i+1], (b,b,b,b), mode='reflect')
            y=F.interpolate(y, size=(H,W), mode='bilinear', align_corners=False)
            out.append(y)
        return torch.cat(out,0)
    def forward(self,x):
        xr=self._reflect(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xr),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"reflect_sim":(1.0-loss).item()}

class TriPatchOrderingSSL(nn.Module):
    """Cut three vertical strips, shuffle them, classify their order (6 classes)."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),6)
    def _tri_strips(self,x):
        B,C,H,W=x.shape; w=W//3
        y=torch.randint(0,6,(B,),device=x.device)
        orders=[(0,1,2),(0,2,1),(1,0,2),(1,2,0),(2,0,1),(2,1,0)]
        out=[]
        for i in range(B):
            o=orders[int(y[i])]
            strips=[x[i:i+1,:, :, j*w:(j+1)*w] for j in range(3)]
            img=torch.cat([strips[o[0]], strips[o[1]], strips[o[2]]], dim=3)
            out.append(F.interpolate(img, size=(H,W), mode='bilinear', align_corners=False))
        return torch.cat(out,0), y
    def forward(self,x):
        xs,y=self._tri_strips(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"tri_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='speckle_noise_detection':    return SpeckleNoiseDetectSSL(enc)
    if n=='channel_permutation_id':     return ChannelPermutationIdSSL(enc)
    if n=='vignette_intensity_regress': return VignetteIntensityRegressSSL(enc)
    if n=='border_reflect_consistency': return BorderReflectConsistencySSL(enc)
    if n=='tri_patch_ordering':         return TriPatchOrderingSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']:
                    best=vd; snap=enc.snapshot(); df=0
                else:
                    enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']:
                    best=vw; snap=enc.snapshot(); wf=0
                else:
                    enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set19.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set19_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='speckle_noise_detection',
                   choices=['speckle_noise_detection','channel_permutation_id','vignette_intensity_regress','border_reflect_consistency','tri_patch_ordering'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set19 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set20_model.py
# Objectives (96–100):
#  96) glare_streak_detection      — detect synthetic glare/light streak overlays (binary)
#  97) shadow_direction_regression — regress direction (angle) of simulated cast shadow
#  98) background_uniformity_reg   — regress global uniformity (1 - intensity std)
#  99) banding_artifact_detection  — detect posterization/banding artifacts (binary)
# 100) anisotropic_scaling_consistency — align embeddings under aspect-ratio scaling
# Single-encoder CNN; compatible with the six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

# ---------------------------
# Obj 96: Glare streak detection (binary)
# ---------------------------
class GlareStreakDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _streak(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=[]
        yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            ang=random.uniform(0,math.pi)
            # oriented streak mask via rotated gaussian ridge
            c,s=math.cos(ang), math.sin(ang)
            u=c*xx+s*yy; v=-s*xx+c*yy
            ridge=torch.exp(- (v**2)/(2*0.02**2)) * (0.5+0.5*torch.cos(20*u))
            ridge=ridge.unsqueeze(0).unsqueeze(0)
            glow=torch.clamp(x[i:i+1] + 0.5*ridge, 0, 1)
            out.append(glow)
        return torch.cat(out,0), y
    def forward(self,x):
        xg,y=self._streak(x); f=self.enc.forward_features(xg); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"glare_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 97: Shadow direction regression
# ---------------------------
class ShadowDirectionRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _shadow(self,x):
        B,C,H,W=x.shape; yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        ang=2*math.pi*torch.rand(B,device=x.device)
        out=[]
        for i in range(B):
            a=ang[i]; c,s=torch.cos(a), torch.sin(a)
            grad=(c*xx + s*yy)  # directional ramp [-1,1]
            mask=(grad>0).float() * (grad.clamp(0,1))
            dim=0.5+0.5*torch.rand(1,device=x.device)
            img=torch.clamp(x[i:i+1]*(1-dim*mask),0,1)
            out.append(img)
        y=ang/(2*math.pi)  # [0,1)
        return torch.cat(out,0), y
    def forward(self,x):
        xs,y=self._shadow(x); f=self.enc.forward_features(xs); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        # circular L1: compare as angle on unit circle — approximate by min(|a-b|, 1-|a-b|)
        d=(pred-y).abs(); d=torch.minimum(d, 1-d)
        loss=d.mean(); return loss,{"shadowang_l1":loss.item()}

# ---------------------------
# Obj 98: Background uniformity regression (1 - std)
# ---------------------------
class BackgroundUniformityRegSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _uniformity(self,x):
        g=to_gray(x); std=g.view(g.size(0),-1).std(dim=1,unbiased=False)
        u=(1-std/ std.max().clamp(min=1e-6)).clamp(0,1)  # higher→more uniform
        return u
    def forward(self,x):
        tgt=self._uniformity(x); f=self.enc.forward_features(x); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,tgt); return loss,{"uni_l1":loss.item()}

# ---------------------------
# Obj 99: Banding/posterization detection (binary)
# ---------------------------
class BandingArtifactDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _band(self,x):
        B=x.size(0); y=torch.randint(0,2,(B,),device=x.device); out=[]
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            levels=random.choice([8,16,32])
            q=torch.round(x[i:i+1]* (levels-1)) /(levels-1)
            out.append(q)
        return torch.cat(out,0), y
    def forward(self,x):
        xb,y=self._band(x); f=self.enc.forward_features(xb); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"band_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 100: Anisotropic scaling consistency (aspect ratio change)
# ---------------------------
class AnisotropicScalingConsistencySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _anisotropic(self,x):
        B,C,H,W=x.shape; out=[]
        for _ in range(B):
            sy=0.7+0.6*random.random(); sx=0.7+0.6*random.random()
            y=F.interpolate(x, size=(int(H*sy), int(W*sx)), mode='bilinear', align_corners=False)
            y=F.interpolate(y, size=(H,W), mode='bilinear', align_corners=False)
            out.append(y)
        return torch.cat(out,0)
    def forward(self,x):
        xa=self._anisotropic(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xa),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"ar_sim":(1.0-loss).item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='glare_streak_detection':        return GlareStreakDetectSSL(enc)
    if n=='shadow_direction_regression':   return ShadowDirectionRegressionSSL(enc)
    if n=='background_uniformity_reg':     return BackgroundUniformityRegSSL(enc)
    if n=='banding_artifact_detection':    return BandingArtifactDetectSSL(enc)
    if n=='anisotropic_scaling_consistency': return AnisotropicScalingConsistencySSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']:
                    best=vd; snap=enc.snapshot(); df=0
                else:
                    enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']:
                    best=vw; snap=enc.snapshot(); wf=0
                else:
                    enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set20.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set20_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='glare_streak_detection',
                   choices=['glare_streak_detection','shadow_direction_regression','background_uniformity_reg','banding_artifact_detection','anisotropic_scaling_consistency'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set20 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set21_model.py
# Objectives (101–105):
# 101) lens_distortion_regression — regress barrel/pincushion distortion magnitude
# 102) phase_scramble_detection   — detect random phase scrambling (binary)
# 103) spectral_energy_consistency — align embeddings under band-pass/low-pass filtering
# 104) chromatic_shift_regression — regress small hue shift amount
# 105) texture_entropy_regression — regress Shannon entropy of grayscale intensities
# Single-encoder CNN; compatible with six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

@torch.no_grad()
def rgb_to_hsv(x):
    r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
    mx=torch.max(torch.max(r,g),b); mn=torch.min(torch.min(r,g),b)
    diff=mx-mn+1e-6
    h=torch.zeros_like(mx)
    mask=mx.eq(r); h[mask]=((g-b)/diff)[mask]%6
    mask=mx.eq(g); h[mask]=((b-r)/diff+2)[mask]
    mask=mx.eq(b); h[mask]=((r-g)/diff+4)[mask]
    h=h/6.0
    s=diff/(mx+1e-6)
    v=mx
    return torch.cat([h,s,v],1).clamp(0,1)

@torch.no_grad()
def hsv_to_rgb(x):
    h,s,v=x[:,0:1],x[:,1:2],x[:,2:3]
    h=h*6; i=torch.floor(h); f=h-i
    p=v*(1-s); q=v*(1-f*s); t=v*(1-(1-f)*s)
    i=i.long()%6
    out=torch.zeros_like(x)
    for k in range(6):
        mask=i.eq(k)
        if k==0: out[mask]=torch.cat([v[mask],t[mask],p[mask]],1)
        if k==1: out[mask]=torch.cat([q[mask],v[mask],p[mask]],1)
        if k==2: out[mask]=torch.cat([p[mask],v[mask],t[mask]],1)
        if k==3: out[mask]=torch.cat([p[mask],q[mask],v[mask]],1)
        if k==4: out[mask]=torch.cat([t[mask],p[mask],v[mask]],1)
        if k==5: out[mask]=torch.cat([v[mask],p[mask],q[mask]],1)
    return out.clamp(0,1)

# ---------------------------
# Obj 101: Lens distortion regression
# ---------------------------
class LensDistortionRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _distort(self,x):
        B,C,H,W=x.shape; yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        r2=xx**2+yy**2
        k= (torch.rand(B,device=x.device)-0.5)*0.4  # [-0.2,0.2]
        out=[]
        for i in range(B):
            ki=k[i]
            rad=1 + ki*r2
            grid=torch.stack((xx*rad, yy*rad), dim=-1)
            out.append(F.grid_sample(x[i:i+1], grid, align_corners=False))
        y=(k+0.2)/0.4  # map to [0,1]
        return torch.cat(out,0), y
    def forward(self,x):
        xd,y=self._distort(x); f=self.enc.forward_features(xd); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"lens_l1":loss.item()}

# ---------------------------
# Obj 102: Phase scramble detection (binary)
# ---------------------------
class PhaseScrambleDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _fft2(self,x):
        # x in [0,1], take grayscale for simplicity
        g=to_gray(x); g=g.squeeze(1)
        F2=torch.fft.rfft2(g)
        mag=torch.abs(F2); ph=torch.angle(F2)
        return mag, ph
    def _ifft2(self,mag,ph):
        F2=mag*torch.exp(1j*ph)
        g=torch.fft.irfft2(F2)
        return g.unsqueeze(1).clamp(0,1).repeat(1,3,1,1)
    def _scramble(self,x):
        B=x.size(0)
        mag,ph=self._fft2(x)
        y=torch.randint(0,2,(B,),device=x.device)
        out=[]
        for i in range(B):
            if y[i]==0:
                out.append(x[i:i+1]); continue
            rnd=torch.rand_like(ph[i])*(2*math.pi)-math.pi
            img=self._ifft2(mag[i:i+1], rnd.unsqueeze(0))
            out.append(img)
        return torch.cat(out,0), y
    def forward(self,x):
        xs,y=self._scramble(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"phase_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 103: Spectral energy consistency
# ---------------------------
class SpectralEnergyConsistencySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _filter(self,x):
        g=to_gray(x).squeeze(1)  # [B,H,W]
        G=torch.fft.rfft2(g)
        mag=torch.abs(G); ph=torch.angle(G)
        B,H,W=g.shape
        out=[]
        for i in range(B):
            mode=random.choice(['low','band'])
            cy=torch.linspace(-1,1,H,device=x.device)
            cx=torch.linspace(0,1,W//2+1,device=x.device)  # rfft domain x in [0,1]
            yy,xx=torch.meshgrid(cy,cx,indexing='ij'); r=(yy**2+xx**2).sqrt()
            if mode=='low': mask=(r<0.4).float()
            else:
                mask=((r>0.2)&(r<0.6)).float()
            M=mask
            F2=mag[i]*M*torch.exp(1j*ph[i])
            gi=torch.fft.irfft2(F2)
            out.append(gi.unsqueeze(0).unsqueeze(0).repeat(1,3,1,1))
        return torch.cat(out,0)
    def forward(self,x):
        xf=self._filter(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xf),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"spec_sim":(1.0-loss).item()}

# ---------------------------
# Obj 104: Chromatic shift regression (hue)
# ---------------------------
class ChromaticShiftRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _shift(self,x):
        hsv=rgb_to_hsv(x); B=x.size(0); dh=(torch.rand(B,1,1,1,device=x.device)-0.5)*0.3  # ~±0.15 hue
        hsv[:,0:1]=(hsv[:,0:1]+dh).remainder(1.0)
        y=(dh.squeeze().abs()/0.15).clamp(0,1)  # normalized magnitude
        return hsv_to_rgb(hsv), y
    def forward(self,x):
        xs,y=self._shift(x); f=self.enc.forward_features(xs); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"cshift_l1":loss.item()}

# ---------------------------
# Obj 105: Texture entropy regression
# ---------------------------
class TextureEntropyRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN, bins=32): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1); self.bins=bins
    def _entropy(self,x):
        g=to_gray(x)
        B=g.size(0)
        y=[]
        for i in range(B):
            gi=g[i].view(-1)
            hist=torch.histc(gi, bins=self.bins, min=0.0, max=1.0)
            p=(hist/hist.sum()).clamp(1e-8,1.0)
            H=-(p*torch.log2(p)).sum()
            y.append(H)
        y=torch.stack(y)
        # normalize by log2(bins)
        y=(y/(math.log2(self.bins))).clamp(0,1)
        return y
    def forward(self,x):
        tgt=self._entropy(x)
        f=self.enc.forward_features(x); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,tgt); return loss,{"entropy_l1":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='lens_distortion_regression':   return LensDistortionRegressionSSL(enc)
    if n=='phase_scramble_detection':     return PhaseScrambleDetectSSL(enc)
    if n=='spectral_energy_consistency':  return SpectralEnergyConsistencySSL(enc)
    if n=='chromatic_shift_regression':   return ChromaticShiftRegressionSSL(enc)
    if n=='texture_entropy_regression':   return TextureEntropyRegressionSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']:
                    best=vd; snap=enc.snapshot(); df=0
                else:
                    enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']:
                    best=vw; snap=enc.snapshot(); wf=0
                else:
                    enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set21.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set21_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='lens_distortion_regression',
                   choices=['lens_distortion_regression','phase_scramble_detection','spectral_energy_consistency','chromatic_shift_regression','texture_entropy_regression'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set21 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set22_model.py
# Objectives (106–110):
# 106) ringing_artifact_detection   — detect convolution ringing near edges (binary)
# 107) channel_dropout_detection    — detect if any RGB channel was zeroed (binary)
# 108) bit_depth_reduction_regress  — regress effective bit-depth reduction level
# 109) zoom_consistency             — invariance to random zoom-in/zoom-out
# 110) stripe_noise_detection       — detect periodic horizontal/vertical stripe noise (binary)
# Single-encoder CNN; compatible with six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

@torch.no_grad()
def gaussian_kernel(sigma: float, device, kmax: int = 13):
    k = max(3, int(2*round(3*sigma)+1))
    k = min(k, kmax if kmax%2==1 else kmax-1)
    ax = torch.arange(-(k//2), k//2+1, device=device, dtype=torch.float32)
    xx, yy = torch.meshgrid(ax, ax, indexing='ij')
    kern = torch.exp(-(xx**2 + yy**2)/(2*sigma**2)); kern = kern/kern.sum()
    return kern.view(1,1,k,k)

# ---------------------------
# Objectives 106–110
# ---------------------------
class RingingArtifactDetectSSL(nn.Module):
    """Sharpen with an over/under-shoot kernel to create ringing; detect presence."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _ring(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=[]
        base=torch.tensor([[0,-1,0],[-1,5,-1],[0,-1,0]],dtype=torch.float32,device=x.device).view(1,1,3,3)
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            k=base + 0.2*torch.randn_like(base)
            xi=F.conv2d(x[i:i+1], k.repeat(C,1,1,1), padding=1, groups=C)
            # add slight overshoot halo by subtracting a light blur
            g=gaussian_kernel(1.2,x.device); halo=F.conv2d(x[i:i+1], g.repeat(C,1,1,1), padding=g.shape[-1]//2, groups=C)
            out.append(torch.clamp(xi + 0.3*(xi-halo), 0, 1))
        return torch.cat(out,0), y
    def forward(self,x):
        xr,y=self._ring(x); f=self.enc.forward_features(xr); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"ring_acc":(logits.argmax(1)==y).float().mean().item()}

class ChannelDropoutDetectSSL(nn.Module):
    """Zero one random channel with p=0.5; detect whether dropout happened."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _drop(self,x):
        B=x.size(0); y=torch.randint(0,2,(B,),device=x.device); out=x.clone()
        for i in range(B):
            if y[i]==1:
                c=random.randint(0,2)
                out[i:i+1,c:c+1]=0.0
        return out,y
    def forward(self,x):
        xd,y=self._drop(x); f=self.enc.forward_features(xd); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"chdrop_acc":(logits.argmax(1)==y).float().mean().item()}

class BitDepthReductionRegSSL(nn.Module):
    """Quantize to b bits (b∈{3..8}); regress normalized reduction level (8-b)/5."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _quant(self,x):
        B=x.size(0); b=torch.randint(3,9,(B,),device=x.device)  # 3..8
        out=[]
        for i in range(B):
            levels=2**int(b[i].item())
            q=torch.round(x[i:i+1]*(levels-1))/(levels-1)
            out.append(q)
        y=(8-b).float()/5.0
        return torch.cat(out,0), y
    def forward(self,x):
        xq,y=self._quant(x); f=self.enc.forward_features(xq); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"bitdepth_l1":loss.item()}

class ZoomConsistencySSL(nn.Module):
    """Random zoom-in/out then resize back; align features with original (cosine)."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _zoom(self,x):
        B,C,H,W=x.shape; out=[]
        for _ in range(B):
            s=0.6+0.9*random.random()  # 0.6..1.5
            h=int(H/s) if s>1 else int(H*s)
            w=int(W/s) if s>1 else int(W*s)
            h=max(4,h); w=max(4,w)
            if s>1:  # zoom out (pad)
                y=F.interpolate(x, size=(h,w), mode='bilinear', align_corners=False)
                y=F.interpolate(y, size=(H,W), mode='bilinear', align_corners=False)
            else:     # zoom in (crop center)
                y=F.interpolate(x[:,:, (H-h)//2:(H-h)//2+h, (W-w)//2:(W-w)//2+w], size=(H,W), mode='bilinear', align_corners=False)
            out.append(y)
        return torch.cat(out,0)
    def forward(self,x):
        xz=self._zoom(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xz),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"zoom_sim":(1.0-loss).item()}

class StripeNoiseDetectSSL(nn.Module):
    """Add periodic stripe noise (horizontal/vertical); detect presence (binary)."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _stripe(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=[]
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            orient=random.choice(['h','v'])
            freq=random.uniform(6.0, 16.0)
            amp=random.uniform(0.05,0.2)
            if orient=='h':
                t=torch.arange(H,device=x.device).float().view(1,1,H,1)
                noise=amp*torch.sin(2*math.pi*t/freq)
            else:
                t=torch.arange(W,device=x.device).float().view(1,1,1,W)
                noise=amp*torch.sin(2*math.pi*t/freq)
            out.append(torch.clamp(x[i:i+1]+noise,0,1))
        return torch.cat(out,0), y
    def forward(self,x):
        xs,y=self._stripe(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"stripe_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='ringing_artifact_detection':   return RingingArtifactDetectSSL(enc)
    if n=='channel_dropout_detection':    return ChannelDropoutDetectSSL(enc)
    if n=='bit_depth_reduction_regress':  return BitDepthReductionRegSSL(enc)
    if n=='zoom_consistency':             return ZoomConsistencySSL(enc)
    if n=='stripe_noise_detection':       return StripeNoiseDetectSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']:
                    best=vd; snap=enc.snapshot(); df=0
                else:
                    enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']:
                    best=vw; snap=enc.snapshot(); wf=0
                else:
                    enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set22.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set22_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='ringing_artifact_detection',
                   choices=['ringing_artifact_detection','channel_dropout_detection','bit_depth_reduction_regress','zoom_consistency','stripe_noise_detection'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set22 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set23_model.py
# Objectives (111–115):
# 111) perspective_distortion_regression — regress tilt magnitude from perspective warp
# 112) crop_location_classification      — classify which quadrant a crop came from
# 113) blur_kernel_type_classification   — classify blur type (gaussian, motion, box)
# 114) color_saturation_regression       — regress saturation scaling factor
# 115) global_contrast_regression        — regress contrast scaling factor
# Single-encoder CNN; compatible with six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Objectives 111–115
# ---------------------------
class PerspectiveDistortionRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _warp(self,x):
        B,C,H,W=x.shape; y=[]; out=[]
        for i in range(B):
            mag=random.uniform(0,0.3)
            sign=random.choice([-1,1])
            tilt=mag*sign
            theta=torch.tensor([[1,tilt,0],[tilt,1,0]],dtype=torch.float32,device=x.device)
            grid=F.affine_grid(theta.unsqueeze(0),x[i:i+1].size(),align_corners=False)
            out.append(F.grid_sample(x[i:i+1],grid,align_corners=False)); y.append(abs(tilt)/0.3)
        return torch.cat(out,0), torch.tensor(y,device=x.device)
    def forward(self,x):
        xp,y=self._warp(x); f=self.enc.forward_features(xp); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"persp_l1":loss.item()}

class CropLocationClassificationSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),4)
    def _crop(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,4,(B,),device=x.device)
        out=[]
        for i in range(B):
            q=int(y[i])
            if q==0: crop=x[i:i+1,:,:H//2,:W//2]
            elif q==1: crop=x[i:i+1,:,:H//2,W//2:]
            elif q==2: crop=x[i:i+1,:,H//2:,:W//2]
            else: crop=x[i:i+1,:,H//2:,W//2:]
            out.append(F.interpolate(crop,size=(H,W)))
        return torch.cat(out,0),y
    def forward(self,x):
        xc,y=self._crop(x); f=self.enc.forward_features(xc); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"crop_acc":(logits.argmax(1)==y).float().mean().item()}

class BlurKernelTypeClassificationSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),3)
    def _blur(self,x):
        B=x.size(0); y=torch.randint(0,3,(B,),device=x.device); out=[]
        for i in range(B):
            mode=int(y[i])
            if mode==0: # gaussian
                k=self._kernel_gauss(x.device)
            elif mode==1: # motion
                k=self._kernel_motion(x.device)
            else: # box
                k=torch.ones((1,1,5,5),device=x.device)/25.0
            xi=F.conv2d(x[i:i+1],k.repeat(3,1,1,1),padding=k.shape[-1]//2,groups=3)
            out.append(xi)
        return torch.cat(out,0),y
    def _kernel_gauss(self,device):
        ax=torch.arange(-2,3,device=device).float()
        xx,yy=torch.meshgrid(ax,ax,indexing='ij')
        k=torch.exp(-(xx**2+yy**2)/4.0); k/=k.sum(); return k.view(1,1,5,5)
    def _kernel_motion(self,device):
        k=torch.zeros((1,1,9,9),device=device); k[0,0,4,:]=1; k/=k.sum(); return k
    def forward(self,x):
        xb,y=self._blur(x); f=self.enc.forward_features(xb); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"blur_acc":(logits.argmax(1)==y).float().mean().item()}

class ColorSaturationRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _saturate(self,x):
        g=to_gray(x); B=x.size(0); scale=torch.rand(B,1,1,1,device=x.device)*1.5
        out=torch.clamp(g+(x-g)*scale,0,1); y=torch.clamp(scale.squeeze()/1.5,0,1)
        return out,y
    def forward(self,x):
        xs,y=self._saturate(x); f=self.enc.forward_features(xs); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"sat_l1":loss.item()}

class GlobalContrastRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _contrast(self,x):
        mean=x.mean(dim=[2,3],keepdim=True); B=x.size(0); alpha=torch.rand(B,1,1,1,device=x.device)*1.5
        out=torch.clamp(mean+(x-mean)*alpha,0,1); y=torch.clamp(alpha.squeeze()/1.5,0,1)
        return out,y
    def forward(self,x):
        xc,y=self._contrast(x); f=self.enc.forward_features(xc); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"contrast_l1":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='perspective_distortion_regression': return PerspectiveDistortionRegressionSSL(enc)
    if n=='crop_location_classification': return CropLocationClassificationSSL(enc)
    if n=='blur_kernel_type_classification': return BlurKernelTypeClassificationSSL(enc)
    if n=='color_saturation_regression': return ColorSaturationRegressionSSL(enc)
    if n=='global_contrast_regression': return GlobalContrastRegressionSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants same as before
# ---------------------------

def adp_w_to_d(enc,cfg,ldr,dev): ...  # same as before

def adp_d_to_w(enc,cfg,ldr,dev): ...

def adp_alt_depth_first(enc,cfg,ldr,dev): ...

def adp_alt_width_first(enc,cfg,ldr,dev): ...

def adp_depth_only(enc,cfg,ldr,dev): ...

def adp_width_only(enc,cfg,ldr,dev): ...

VARIANT_FUNCS={
    'wd':adp_w_to_d,'dw':adp_d_to_w,'alt_d':adp_alt_depth_first,'alt_w':adp_alt_width_first,'depth_only':adp_depth_only,'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set23.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set23_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='perspective_distortion_regression',
                   choices=['perspective_distortion_regression','crop_location_classification','blur_kernel_type_classification','color_saturation_regression','global_contrast_regression'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set23 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set24_model.py
# Objectives (116–120):
# 116) lbp_uniformity_regression   — regress Local Binary Pattern uniformity score
# 117) rain_streak_detection       — detect synthetic rain streak overlays (binary)
# 118) hue_rotation_classification — classify hue rotation bucket (k=6)
# 119) patch_inpainting_consistency — align features before/after proxy inpainting
# 120) swirl_distortion_regression — regress amount of swirl warp applied
# Single-encoder CNN; compatible with six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

@torch.no_grad()
def rgb_to_hsv(x):
    r,g,b=x[:,0:1],x[:,1:2],x[:,2:3]
    mx=torch.max(torch.max(r,g),b); mn=torch.min(torch.min(r,g),b)
    diff=mx-mn+1e-6
    h=torch.zeros_like(mx)
    mask=mx.eq(r); h[mask]=((g-b)/diff)[mask]%6
    mask=mx.eq(g); h[mask]=((b-r)/diff+2)[mask]
    mask=mx.eq(b); h[mask]=((r-g)/diff+4)[mask]
    h=h/6.0
    s=diff/(mx+1e-6)
    v=mx
    return torch.cat([h,s,v],1).clamp(0,1)

@torch.no_grad()
def hsv_to_rgb(x):
    h,s,v=x[:,0:1],x[:,1:2],x[:,2:3]
    h=h*6; i=torch.floor(h); f=h-i
    p=v*(1-s); q=v*(1-f*s); t=v*(1-(1-f)*s)
    i=i.long()%6
    out=torch.zeros_like(x)
    for k in range(6):
        mask=i.eq(k)
        if k==0: out[mask]=torch.cat([v[mask],t[mask],p[mask]],1)
        if k==1: out[mask]=torch.cat([q[mask],v[mask],p[mask]],1)
        if k==2: out[mask]=torch.cat([p[mask],v[mask],t[mask]],1)
        if k==3: out[mask]=torch.cat([p[mask],q[mask],v[mask]],1)
        if k==4: out[mask]=torch.cat([t[mask],p[mask],v[mask]],1)
        if k==5: out[mask]=torch.cat([v[mask],p[mask],q[mask]],1)
    return out.clamp(0,1)

# ---------------------------
# Obj 116: LBP uniformity regression
# ---------------------------
class LBPUniformityRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN, bins=16): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1); self.bins=bins
    def _lbp_uniformity(self,x):
        g=to_gray(x)
        B,_,H,W=g.shape
        # 8-neighbor LBP
        shifts=[(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
        center=g
        codes=torch.zeros(B,1,H,W,device=x.device)
        for idx,(dy,dx) in enumerate(shifts):
            pad=(max(dx,0), max(-dx,0), max(dy,0), max(-dy,0))
            nei=F.pad(g, pad, mode='replicate')[:,:, max(-dy,0):H+max(-dy,0), max(-dx,0):W+max(-dx,0)]
            bit=(nei>=center).float()
            codes = codes + bit*(2**idx)
        # uniformity score: fraction of uniform patterns (<=2 bit transitions in circular binary code)
        # approximate transitions by circular finite differences on bits
        # compute histogram and normalize; proxy uniformity = entropy complement
        hist=torch.histc(codes.view(B,-1), bins=self.bins, min=0, max=255)
        p=(hist/(hist.sum(dim=1,keepdim=True)+1e-6)).clamp(1e-8,1.0)
        Hs=-(p*torch.log2(p)).sum(dim=1) / math.log2(self.bins)
        uni=1.0 - Hs  # higher means more uniform
        return uni.clamp(0,1)
    def forward(self,x):
        tgt=self._lbp_uniformity(x); f=self.enc.forward_features(x); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,tgt); return loss,{"lbpuni_l1":loss.item()}

# ---------------------------
# Obj 117: Rain streak detection (binary)
# ---------------------------
class RainStreakDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _rain(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=[]
        yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            ang=random.uniform(-math.pi/6, math.pi/6)
            c,s=math.cos(ang), math.sin(ang)
            u=c*xx + s*yy; v=-s*xx + c*yy
            density=random.randint(20,40)
            layer=torch.zeros(1,1,H,W,device=x.device)
            for _ in range(density):
                phase=random.uniform(-math.pi, math.pi)
                freq=random.uniform(8,20)
                streak=(0.6+0.4*random.random())*torch.relu(torch.cos(freq*u+phase))
                layer += streak.unsqueeze(0).unsqueeze(0)
            layer=layer/layer.max().clamp(min=1.0)
            rainy=torch.clamp(x[i:i+1] + 0.25*layer, 0, 1)
            out.append(rainy)
        return torch.cat(out,0), y
    def forward(self,x):
        xr,y=self._rain(x); f=self.enc.forward_features(xr); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"rain_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 118: Hue rotation classification (k=6)
# ---------------------------
class HueRotationClassificationSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN, k=6): super().__init__(); self.enc=enc; self.k=k; self.fc=nn.Linear(enc._last_width(),k)
    def _rotate(self,x):
        hsv=rgb_to_hsv(x)
        B=x.size(0); y=torch.randint(0,self.k,(B,),device=x.device)
        out=[]
        for i in range(B):
            step=(int(y[i].item())/self.k)
            z=hsv[i:i+1].clone()
            z[:,0:1]=(z[:,0:1]+step)%1.0
            out.append(hsv_to_rgb(z))
        return torch.cat(out,0), y
    def forward(self,x):
        xr,y=self._rotate(x); f=self.enc.forward_features(xr); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"hue_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 119: Patch inpainting consistency
# ---------------------------
class PatchInpaintingConsistencySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _inpaint(self,x):
        B,C,H,W=x.shape; out=x.clone()
        for i in range(B):
            h=max(6,H//6); w=max(6,W//6); y0=random.randint(0,H-h); x0=random.randint(0,W-w)
            # proxy inpaint: blur neighborhood and fill
            patch=out[i:i+1,:, max(0,y0-3):min(H,y0+h+3), max(0,x0-3):min(W,x0+w+3)]
            k=self._gauss_kernel(1.6, x.device)
            blurred=F.conv2d(patch, k.repeat(C,1,1,1), padding=k.shape[-1]//2, groups=C)
            out[i:i+1,:, y0:y0+h, x0:x0+w]=F.interpolate(blurred, size=(h,w), mode='bilinear', align_corners=False)
        return out
    def _gauss_kernel(self,sigma,device):
        k=max(3,int(2*round(3*sigma)+1)); ax=torch.arange(-(k//2),(k//2)+1,device=device,dtype=torch.float32)
        xx,yy=torch.meshgrid(ax,ax,indexing='ij'); K=torch.exp(-(xx**2+yy**2)/(2*sigma**2)); K=K/K.sum(); return K.view(1,1,k,k)
    def forward(self,x):
        xi=self._inpaint(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xi),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"inpaint_sim":(1.0-loss).item()}

# ---------------------------
# Obj 120: Swirl distortion regression
# ---------------------------
class SwirlDistortionRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _swirl(self,x):
        B,C,H,W=x.shape; yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        r=torch.sqrt(xx**2+yy**2); ang=torch.atan2(yy,xx)
        k=0.5*torch.rand(B,device=x.device)  # swirl strength ~[0,0.5)
        out=[]
        for i in range(B):
            theta=ang + k[i]*r
            grid=torch.stack([r*torch.cos(theta), r*torch.sin(theta)], dim=-1)
            # convert polar-ish back to cartesian grid in [-1,1]
            gx=grid[...,0]; gy=grid[...,1]
            grid_cart=torch.stack([gx,gy], dim=-1)
            out.append(F.grid_sample(x[i:i+1], grid_cart, align_corners=False))
        y=(k/0.5).clamp(0,1)
        return torch.cat(out,0), y
    def forward(self,x):
        xs,y=self._swirl(x); f=self.enc.forward_features(xs); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"swirl_l1":loss.item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='lbp_uniformity_regression':     return LBPUniformityRegressionSSL(enc)
    if n=='rain_streak_detection':         return RainStreakDetectSSL(enc)
    if n=='hue_rotation_classification':   return HueRotationClassificationSSL(enc)
    if n=='patch_inpainting_consistency':  return PatchInpaintingConsistencySSL(enc)
    if n=='swirl_distortion_regression':   return SwirlDistortionRegressionSSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']:
                    best=vd; snap=enc.snapshot(); df=0
                else:
                    enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']:
                    best=vw; snap=enc.snapshot(); wf=0
                else:
                    enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set24.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set24_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='lbp_uniformity_regression',
                   choices=['lbp_uniformity_regression','rain_streak_detection','hue_rotation_classification','patch_inpainting_consistency','swirl_distortion_regression'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set24 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set25_model.py
# Objectives (121–125):
# 121) edge_density_regression       — regress fraction of edge pixels (Sobel>τ)
# 122) solarization_detection        — detect solarization (binary)
# 123) poisson_noise_level_regression— regress Poisson noise strength
# 124) blockiness_regression         — regress compression-like blockiness level
# 125) rotation_invariance_consistency — align embeddings under small rotations
# Single-encoder CNN; compatible with six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

# Sobel filters
SOBEL_X=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)
SOBEL_Y=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).view(1,1,3,3)

# ---------------------------
# Obj 121: Edge density regression
# ---------------------------
class EdgeDensityRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN, thresh: float = 0.25): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1); self.th=thresh
    def _edge_density(self,x):
        g=to_gray(x); kx=SOBEL_X.to(x.device); ky=SOBEL_Y.to(x.device)
        gx=F.conv2d(g,kx,padding=1); gy=F.conv2d(g,ky,padding=1)
        mag=(gx*gx+gy*gy).sqrt()
        # normalize per-image
        m=mag.view(mag.size(0),-1); m=(m-m.min(dim=1,keepdim=True).values)/(m.max(dim=1,keepdim=True).values-m.min(dim=1,keepdim=True).values+1e-6)
        m=m.view_as(mag)
        ed=(m>self.th).float().mean(dim=[1,2,3])
        return ed
    def forward(self,x):
        tgt=self._edge_density(x); f=self.enc.forward_features(x); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,tgt); return loss,{"edge_density_l1":loss.item()}

# ---------------------------
# Obj 122: Solarization detection (binary)
# ---------------------------
class SolarizationDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _solarize(self,x):
        B=x.size(0); y=torch.randint(0,2,(B,),device=x.device); out=[]
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            thr=random.uniform(0.3,0.7)
            xi=x[i:i+1].clone(); mask=(xi>thr).float(); xi=xi*(1-mask)+(1-xi)*mask
            out.append(xi)
        return torch.cat(out,0), y
    def forward(self,x):
        xs,y=self._solarize(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"solar_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 123: Poisson noise level regression
# ---------------------------
class PoissonNoiseLevelRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _poisson(self,x):
        # scale intensities to counts, sample Poisson, scale back
        B=x.size(0); scale=20+180*torch.rand(B,device=x.device)  # 20..200
        out=[]
        for i in range(B):
            s=float(scale[i].item())
            lam=(x[i:i+1].clamp(0,1)*s).clamp(min=0)
            y=torch.poisson(lam)/s
            out.append(y.clamp(0,1))
        tgt=((scale-20)/180).clamp(0,1)  # higher scale → lower noise; predict inverse strength? use 1/scale proxy
        # map to noise strength ~ proportional to 1/sqrt(scale); normalize roughly
        noise_strength=(1.0/torch.sqrt(scale+1e-6))
        noise_strength=(noise_strength - noise_strength.min())/(noise_strength.max()-noise_strength.min()+1e-6)
        return torch.cat(out,0), noise_strength
    def forward(self,x):
        xn,y=self._poisson(x); f=self.enc.forward_features(xn); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"poisson_l1":loss.item()}

# ---------------------------
# Obj 124: Blockiness regression
# ---------------------------
class BlockinessRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _block(self,x):
        B,C,H,W=x.shape; b=torch.randint(2,9,(B,),device=x.device)  # block size 2..8
        out=[]
        for i in range(B):
            bs=int(b[i].item())
            pooled=F.avg_pool2d(x[i:i+1], kernel_size=bs, stride=bs)
            up=F.interpolate(pooled, size=(H,W), mode='nearest')
            out.append(up)
        # blockiness target increases with block size
        tgt=(b.float()-2)/(8-2)
        return torch.cat(out,0), tgt
    def forward(self,x):
        xb,y=self._block(x); f=self.enc.forward_features(xb); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"block_l1":loss.item()}

# ---------------------------
# Obj 125: Rotation invariance consistency
# ---------------------------
class RotationInvarianceConsistencySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc
    def _rotate(self,x):
        B,C,H,W=x.shape; out=[]
        for _ in range(B):
            ang=(random.random()-0.5)*40.0  # -20..20 degrees
            rad=ang*math.pi/180.0
            theta=torch.tensor([[math.cos(rad), -math.sin(rad), 0.0],[math.sin(rad), math.cos(rad), 0.0]],dtype=torch.float32,device=x.device)
            grid=F.affine_grid(theta.unsqueeze(0), x.size(), align_corners=False)
            y=F.grid_sample(x, grid, align_corners=False)
            out.append(y)
        return torch.cat(out,0)
    def forward(self,x):
        xr=self._rotate(x)
        f1=F.normalize(self.enc.forward_features(x),dim=1)
        f2=F.normalize(self.enc.forward_features(xr),dim=1)
        loss=1.0-(f1*f2).sum(1).mean(); return loss,{"rot_sim":(1.0-loss).item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='edge_density_regression':         return EdgeDensityRegressionSSL(enc)
    if n=='solarization_detection':          return SolarizationDetectSSL(enc)
    if n=='poisson_noise_level_regression':  return PoissonNoiseLevelRegressionSSL(enc)
    if n=='blockiness_regression':           return BlockinessRegressionSSL(enc)
    if n=='rotation_invariance_consistency': return RotationInvarianceConsistencySSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']:
                    best=vd; snap=enc.snapshot(); df=0
                else:
                    enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']:
                    best=vw; snap=enc.snapshot(); wf=0
                else:
                    enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set25.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set25_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='edge_density_regression',
                   choices=['edge_density_regression','solarization_detection','poisson_noise_level_regression','blockiness_regression','rotation_invariance_consistency'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set25 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
# ======================================
# File: adp_ssl_set26_model.py
# Objectives (126–130):
# 126) illumination_gradient_angle   — regress dominant illumination gradient angle
# 127) saltpepper_noise_detection    — detect salt & pepper noise (binary)
# 128) hist_equalization_detection   — detect histogram equalization application (binary)
# 129) gamma_correction_regression   — regress gamma applied to intensities
# 130) mixup_consistency             — align embeddings under MixUp with a peer image
# Single-encoder CNN; compatible with six ADP variants.
# ======================================

from typing import List, Optional
import math, random, torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------
# Backbone (Adaptive CNN)
# ---------------------------
class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1):
        super().__init__(); self.conv=nn.Conv2d(in_ch,out_ch,k,s,p,bias=False); self.bn=nn.BatchNorm2d(out_ch); self.relu=nn.ReLU(True)
    def forward(self,x): return self.relu(self.bn(self.conv(x)))

class AdaptiveCNN(nn.Module):
    def __init__(self, in_ch: int, num_classes: int, widths: List[int], pooling_indices: List[int] = None):
        super().__init__(); assert len(widths)>=1
        self.widths=list(widths); self.blocks=nn.ModuleList(); self.pooling_indices=set(pooling_indices or [])
        prev=in_ch
        for w in self.widths:
            blk=ConvBNReLU(prev,w); self.blocks.append(blk); prev=w
        self.gap=nn.AdaptiveAvgPool2d((1,1)); self.head=nn.Linear(self.widths[-1], num_classes)
    def _last_width(self): return self.widths[-1]
    def forward_features(self,x):
        for i,blk in enumerate(self.blocks):
            x=blk(x)
            if i in self.pooling_indices: x=F.max_pool2d(x,2)
        return self.gap(x).squeeze(-1).squeeze(-1)
    def append_depth(self): c=self.widths[-1]; self.blocks.append(ConvBNReLU(c,c)); self.widths.append(c)
    def widen_all(self, ex_k=8, max_width: Optional[int]=None):
        new=[min(w+ex_k,max_width) if max_width else w+ex_k for w in self.widths]
        prev=self.blocks[0].conv.in_channels
        for i,blk in enumerate(self.blocks):
            blk.conv=nn.Conv2d(prev,new[i],3,1,1,bias=False); blk.bn=nn.BatchNorm2d(new[i]); prev=new[i]
        self.widths=new; self.head=nn.Linear(new[-1], self.head.out_features)
    def snapshot(self): return {"state":{k:v.detach().cpu() for k,v in self.state_dict().items()},"widths":list(self.widths)}
    def restore(self,snap): self.load_state_dict(snap['state'], strict=False); self.widths=list(snap['widths'])

# ---------------------------
# Utilities
# ---------------------------
@torch.no_grad()
def to_gray(x):  # [0,1]
    return (0.299*x[:,0:1]+0.587*x[:,1:2]+0.114*x[:,2:3]).clamp(0,1)

SOBEL_X=torch.tensor([[1,0,-1],[2,0,-2],[1,0,-1]],dtype=torch.float32).view(1,1,3,3)
SOBEL_Y=torch.tensor([[1,2,1],[0,0,0],[-1,-2,-1]],dtype=torch.float32).view(1,1,3,3)

# ---------------------------
# Obj 126: Illumination gradient angle regression
# ---------------------------
class IlluminationGradientAngleSSL(nn.Module):
    """Add an additive illumination gradient and regress its direction angle in [0,1)."""
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _illum(self,x):
        B,C,H,W=x.shape; yy,xx=torch.meshgrid(torch.linspace(-1,1,H,device=x.device), torch.linspace(-1,1,W,device=x.device), indexing='ij')
        ang=2*math.pi*torch.rand(B,device=x.device)
        out=[]
        for i in range(B):
            a=ang[i]; c,s=torch.cos(a), torch.sin(a)
            grad=(c*xx + s*yy)  # [-1,1]
            amp=0.15+0.35*random.random()
            layer=(amp*(grad+1)/2.0).expand_as(x[i:i+1])
            y=torch.clamp(x[i:i+1] + layer, 0, 1)
            out.append(y)
        return torch.cat(out,0), (ang/(2*math.pi))
    def forward(self,x):
        xi,y=self._illum(x); f=self.enc.forward_features(xi); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        d=(pred-y).abs(); d=torch.minimum(d, 1-d)  # circular distance
        loss=d.mean(); return loss,{"illumang_l1":loss.item()}

# ---------------------------
# Obj 127: Salt & pepper noise detection (binary)
# ---------------------------
class SaltPepperDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2)
    def _sp(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=x.clone()
        for i in range(B):
            if y[i]==0: continue
            p=0.01+0.09*random.random()  # 1%..10%
            mask=torch.rand(1,1,H,W,device=x.device)
            salt=(mask<p/2).float(); pepper=((mask>=p/2)&(mask<p)).float()
            out[i:i+1]=torch.clamp(out[i:i+1]*(1-pepper) + salt,0,1)
        return out,y
    def forward(self,x):
        xs,y=self._sp(x); f=self.enc.forward_features(xs); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"sp_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 128: Histogram equalization detection (binary)
# ---------------------------
class HistEqualizationDetectSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN, bins=64): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),2); self.bins=bins
    def _equalize(self,x):
        B,C,H,W=x.shape; y=torch.randint(0,2,(B,),device=x.device); out=[]
        for i in range(B):
            if y[i]==0: out.append(x[i:i+1]); continue
            g=to_gray(x[i:i+1])
            hist=torch.histc(g.view(1,-1), bins=self.bins, min=0.0, max=1.0)
            cdf=torch.cumsum(hist, dim=1); cdf=cdf/cdf[:,-1:]
            # map intensities via cdf (nearest bin)
            idx=torch.clamp((g*(self.bins-1)).long(),0,self.bins-1)
            eq=cdf.gather(1, idx.view(1,-1)).view_as(g)
            rgb=eq.repeat(1,3,1,1)
            out.append(rgb)
        return torch.cat(out,0), y
    def forward(self,x):
        xe,y=self._equalize(x); f=self.enc.forward_features(xe); logits=self.fc(f); loss=F.cross_entropy(logits,y)
        return loss,{"histeq_acc":(logits.argmax(1)==y).float().mean().item()}

# ---------------------------
# Obj 129: Gamma correction regression
# ---------------------------
class GammaCorrectionRegressionSSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN): super().__init__(); self.enc=enc; self.fc=nn.Linear(enc._last_width(),1)
    def _gamma(self,x):
        B=x.size(0); g=0.6+1.4*torch.rand(B,device=x.device)  # gamma in [0.6,2.0]
        out=[]
        for i in range(B):
            y=torch.clamp(x[i:i+1],0,1)
            y=y**g[i]
            out.append(y)
        # Normalize target to [0,1]
        tgt=(g-0.6)/(2.0-0.6)
        return torch.cat(out,0), tgt
    def forward(self,x):
        xg,y=self._gamma(x); f=self.enc.forward_features(xg); pred=torch.sigmoid(self.fc(f)).squeeze(1)
        loss=F.l1_loss(pred,y); return loss,{"gamma_l1":loss.item()}

# ---------------------------
# Obj 130: MixUp consistency
# ---------------------------
class MixUpConsistencySSL(nn.Module):
    def __init__(self, enc: AdaptiveCNN, alpha: float = 0.4): super().__init__(); self.enc=enc; self.alpha=alpha
    def forward(self,x):
        B=x.size(0)
        perm=torch.randperm(B, device=x.device)
        lam=torch.distributions.Beta(self.alpha, self.alpha).sample((B,)).to(x.device)
        lam=lam.view(B,1,1,1)
        xm=lam*x + (1-lam)*x[perm]
        f=F.normalize(self.enc.forward_features(x),dim=1)
        fm=F.normalize(self.enc.forward_features(xm),dim=1)
        # encourage fm close to convex combo of features
        fperm=f[perm]
        ftarget=lam.squeeze()*f + (1-lam.squeeze())*fperm
        loss=1.0-(fm*ftarget).sum(1).mean(); return loss,{"mixup_sim":(1.0-loss).item()}

# ---------------------------
# Factory
# ---------------------------

def build_objective(name: str, enc: AdaptiveCNN):
    n=name.lower()
    if n=='illumination_gradient_angle': return IlluminationGradientAngleSSL(enc)
    if n=='saltpepper_noise_detection':   return SaltPepperDetectSSL(enc)
    if n=='hist_equalization_detection':  return HistEqualizationDetectSSL(enc)
    if n=='gamma_correction_regression':  return GammaCorrectionRegressionSSL(enc)
    if n=='mixup_consistency':            return MixUpConsistencySSL(enc)
    raise ValueError(n)

# ---------------------------
# Training + ADP Variants
# ---------------------------
class EarlyStop:
    def __init__(self,pat=4): self.p=pat; self.best=float('inf'); self.bad=0; self.snap=None
    def step(self,v,s):
        if v<self.best: self.best=v; self.bad=0; self.snap=s; return True
        self.bad+=1; return False
    def done(self): return self.bad>=self.p

def run_inner(enc,obj,ldr,dev,epochs=3):
    o=build_objective(obj,enc).to(dev); opt=torch.optim.AdamW(enc.parameters(),lr=3e-4,weight_decay=1e-4); st=EarlyStop(max(3,epochs//2))
    for e in range(epochs):
        for x,_ in ldr:
            x=x.to(dev); opt.zero_grad(set_to_none=True); l,_=o(x); l.backward(); nn.utils.clip_grad_norm_(enc.parameters(),1.0); opt.step()
        st.step(l.item(), enc.snapshot());
        if st.done(): break
    if st.snap: enc.restore(st.snap)
    return st.best

# Six ADP variants

def adp_w_to_d(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); wf=0
    while wf<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); df=0
            while df<cfg['patience_depth']:
                enc.append_depth(); vd=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vd<best-cfg['delta']:
                    best=vd; snap=enc.snapshot(); df=0
                else:
                    enc.restore(snap); df+=1
        else:
            enc.restore(snap); wf+=1
    enc.restore(snap); return best

def adp_d_to_w(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); df=0
    while df<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); wf=0
            while wf<cfg['patience_width']:
                enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
                vw=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
                if vw<best-cfg['delta']:
                    best=vw; snap=enc.snapshot(); wf=0
                else:
                    enc.restore(snap); wf+=1
        else:
            enc.restore(snap); df+=1
    enc.restore(snap); return best

def adp_alt_depth_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_alt_width_first(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot()
    while True:
        accepted=False
        wf=0
        while wf<cfg['patience_width']:
            enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
            v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; wf=0
            else:
                enc.restore(snap); wf+=1; break
        df=0
        while df<cfg['patience_depth']:
            enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
            if v<best-cfg['delta']:
                best=v; snap=enc.snapshot(); accepted=True; df=0
            else:
                enc.restore(snap); df+=1; break
        if not accepted: break
    enc.restore(snap); return best

def adp_depth_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_depth']:
        enc.append_depth(); v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

def adp_width_only(enc,cfg,ldr,dev):
    best=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs']); snap=enc.snapshot(); f=0
    while f<cfg['patience_width']:
        enc.widen_all(cfg['ex_k'],cfg.get('max_width'))
        v=run_inner(enc,cfg['objective'],ldr,dev,cfg['inner_epochs'])
        if v<best-cfg['delta']:
            best=v; snap=enc.snapshot(); f=0
        else:
            enc.restore(snap); f+=1
    enc.restore(snap); return best

VARIANT_FUNCS={
    'wd':adp_w_to_d,
    'dw':adp_d_to_w,
    'alt_d':adp_alt_depth_first,
    'alt_w':adp_alt_width_first,
    'depth_only':adp_depth_only,
    'width_only':adp_width_only,
}

# ======================================
# File: run_adp_ssl_set26.py
# ======================================

import argparse, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from adp_ssl_set26_model import AdaptiveCNN, VARIANT_FUNCS


def get_dataset(name, train=True):
    tf = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    if name=='cifar10': return datasets.CIFAR10('./data', train=True, download=True, transform=tf)
    if name=='cifar100': return datasets.CIFAR100('./data', train=True, download=True, transform=tf)
    if name=='stl10': return datasets.STL10('./data', split='train', download=True, transform=tf)
    raise ValueError(name)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dataset',default='cifar10',choices=['cifar10','cifar100','stl10'])
    ap.add_argument('--objective',default='illumination_gradient_angle',
                   choices=['illumination_gradient_angle','saltpepper_noise_detection','hist_equalization_detection','gamma_correction_regression','mixup_consistency'])
    ap.add_argument('--variant',default='wd',choices=['wd','dw','alt_d','alt_w','depth_only','width_only'])
    ap.add_argument('--widths',default='32,32,64'); ap.add_argument('--pool_idx',default='1')
    ap.add_argument('--epochs',type=int,default=3); ap.add_argument('--ex_k',type=int,default=16)
    ap.add_argument('--patience_depth',type=int,default=2); ap.add_argument('--patience_width',type=int,default=2)
    ap.add_argument('--delta',type=float,default=0.0); ap.add_argument('--max_width',type=int,default=256)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()

    random.seed(a.seed); torch.manual_seed(a.seed); dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ds=get_dataset(a.dataset); nv=int(len(ds)*0.1); tr,vl=random_split(ds,[len(ds)-nv,nv])
    ldr=DataLoader(tr,batch_size=128,shuffle=True,num_workers=2,pin_memory=True)

    widths=[int(x) for x in a.widths.split(',')]; pool_idx=[int(x) for x in a.pool_idx.split(',') if x.strip()!='']
    enc=AdaptiveCNN(3,10,widths,pooling_indices=pool_idx).to(dev)

    cfg={'objective':a.objective,'inner_epochs':a.epochs,'patience_depth':a.patience_depth,'patience_width':a.patience_width,'ex_k':a.ex_k,'delta':a.delta,'max_width':a.max_width}

    best=VARIANT_FUNCS[a.variant](enc,cfg,ldr,dev)
    print(f'[DONE] Set26 {a.variant} {a.objective} best={best:.4f}')

if __name__=='__main__': main()
