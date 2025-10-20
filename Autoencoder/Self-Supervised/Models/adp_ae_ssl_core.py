
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, bias=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class DeconvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, outpad=0, bias=True):
        super().__init__()
        if p is None: p = k // 2
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, k, s, p, output_padding=outpad, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.deconv(x)))

class AutoencoderSSL(nn.Module):
    def __init__(self, in_ch, widths: List[int], pooling_indices: List[int], bias=True, proj_dim=None):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.pooling_indices = sorted(list(set(pooling_indices)))
        self.bias = bias
        enc, c_in = [], in_ch
        for c_out in widths:
            enc.append(ConvBNReLU(c_in, c_out, bias=bias)); c_in = c_out
        self.encoder = nn.ModuleList(enc)
        self._pools_here = [i in self.pooling_indices for i in range(len(widths))]
        self.pool = nn.MaxPool2d(2,2)
        self.gap = nn.AdaptiveAvgPool2d((1,1))
        hidden = widths[-1]
        pdim = hidden if proj_dim is None else proj_dim
        self.projector = nn.Linear(hidden, pdim, bias=True)
        dec, c_in = [], widths[-1]
        for i in range(len(widths)-1, -1, -1):
            c_out = widths[i-1] if i-1 >= 0 else in_ch
            stride = 2 if self._pools_here[i] else 1
            outpad = 1 if stride == 2 else 0
            dec.append(DeconvBNReLU(c_in, c_out, s=stride, outpad=outpad, bias=bias)); c_in = c_out
        self.decoder = nn.ModuleList(dec)
        self.recon_head = nn.Conv2d(in_ch, in_ch, 3, 1, 1, bias=True)
    @property
    def widths(self): return [blk.bn.num_features for blk in self.encoder]
    def forward(self, x):
        h = x
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            if self._pools_here[i]: h = self.pool(h)
        z = self.gap(h).flatten(1); z = self.projector(z)
        h_dec = h
        for blk in self.decoder: h_dec = blk(h_dec)
        rec = self.recon_head(h_dec)
        return rec, z
    def append_depth(self):
        last_c = self.encoder[-1].bn.num_features
        self.encoder.append(ConvBNReLU(last_c, last_c, bias=self.bias))
        self._pools_here.append(False)
        stride = 2 if self._pools_here[-1] else 1
        outpad = 1 if stride == 2 else 0
        self.decoder.insert(0, DeconvBNReLU(last_c, last_c, s=stride, outpad=outpad, bias=self.bias))
    def widen_all(self, ex_k: int):
        if ex_k <= 0: return
        prev = self.in_ch
        for enc in self.encoder:
            old_out = enc.bn.num_features; new_out = old_out + ex_k
            _resize_conv2d_(enc.conv, prev, new_out)
            _resize_bn2d_(enc.bn, new_out)
            prev = new_out
        _resize_linear_(self.projector, self.projector.in_features + ex_k, self.projector.out_features)
        self._rebuild_decoder([blk.bn.num_features for blk in self.encoder])
    def _rebuild_decoder(self, enc_widths: List[int]):
        old_dec = self.decoder; new_dec = nn.ModuleList(); c_in = enc_widths[-1]
        for i in range(len(enc_widths)-1, -1, -1):
            c_out = enc_widths[i-1] if i-1 >= 0 else self.in_ch
            stride = 2 if self._pools_here[i] else 1; outpad = 1 if stride==2 else 0
            nb = DeconvBNReLU(c_in, c_out, s=stride, outpad=outpad, bias=self.bias)
            new_dec.append(nb); c_in = c_out
        for nb, ob in zip(new_dec, old_dec):
            _resize_convtranspose2d_(nb.deconv, ob.deconv.in_channels, ob.deconv.out_channels)
            _overlap_copy_(nb.deconv.weight.data, ob.deconv.weight.data)
            if nb.deconv.bias is not None and ob.deconv.bias is not None:
                _overlap_copy_(nb.deconv.bias.data, ob.deconv.bias.data)
            _resize_bn2d_(nb.bn, nb.bn.num_features)
            _overlap_copy_(nb.bn.weight.data, ob.bn.weight.data)
            _overlap_copy_(nb.bn.bias.data, ob.bn.bias.data)
            _overlap_copy_(nb.bn.running_mean, ob.bn.running_mean)
            _overlap_copy_(nb.bn.running_var, ob.bn.running_var)
        self.decoder = new_dec
    def total_neurons(self):
        enc_ch = sum(blk.bn.num_features for blk in self.encoder)
        dec_ch = sum(blk.bn.num_features for blk in self.decoder)
        return enc_ch + dec_ch + self.projector.in_features

def _overlap_copy_(dst, src):
    dims = [min(a, b) for a, b in zip(dst.shape, src.shape)]
    dst[tuple(slice(0,d) for d in dims)].copy_(src[tuple(slice(0,d) for d in dims)])

def _resize_conv2d_(conv, in_ch, out_ch):
    old_w = conv.weight.data.clone(); old_b = conv.bias.data.clone() if conv.bias is not None else None
    k_h, k_w = conv.kernel_size; device = conv.weight.device
    conv.in_channels = in_ch; conv.out_channels = out_ch
    conv.weight = nn.Parameter(torch.empty(out_ch, in_ch, k_h, k_w, device=device))
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu'); _overlap_copy_(conv.weight.data, old_w)
    if conv.bias is not None:
        conv.bias = nn.Parameter(torch.zeros(out_ch, device=device))
        if old_b is not None: _overlap_copy_(conv.bias.data, old_b)

def _resize_convtranspose2d_(deconv, in_ch, out_ch):
    old_w = deconv.weight.data.clone(); old_b = deconv.bias.data.clone() if deconv.bias is not None else None
    k_h, k_w = deconv.kernel_size; device = deconv.weight.device
    deconv.in_channels = in_ch; deconv.out_channels = out_ch
    deconv.weight = nn.Parameter(torch.empty(in_ch, out_ch, k_h, k_w, device=device))
    nn.init.kaiming_normal_(deconv.weight, nonlinearity='relu'); _overlap_copy_(deconv.weight.data, old_w)
    if deconv.bias is not None:
        deconv.bias = nn.Parameter(torch.zeros(out_ch, device=device))
        if old_b is not None: _overlap_copy_(deconv.bias.data, old_b)

def _resize_bn2d_(bn, out_ch):
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

def _resize_linear_(fc, in_f, out_f):
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

class TwoViewCIFAR10(Dataset):
    def __init__(self, root, train, t1, t2, download):
        self.base = datasets.CIFAR10(root=root, train=train, transform=None, download=download)
        self.t1, self.t2 = t1, t2
    def __len__(self): return len(self.base)
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.t1(img), self.t2(img)

def make_cifar10_ssl_loaders(data_root, batch_size, num_workers=4, val_split=0.1, download=True, seed=0, two_views=True):
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    aug = transforms.Compose([transforms.RandomResizedCrop(32, scale=(0.6,1.0)),
                              transforms.RandomHorizontalFlip(),
                              transforms.ColorJitter(0.2,0.2,0.2,0.1),
                              transforms.ToTensor(), transforms.Normalize(mean, std)])
    eval_tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    if two_views:
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
    else:
        ds_full = datasets.CIFAR10(root=data_root, train=True, transform=aug, download=download)
        n_val = int(len(ds_full)*val_split); n_train = len(ds_full)-n_val
        g = torch.Generator().manual_seed(seed)
        ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=g)
        ds_test = datasets.CIFAR10(root=data_root, train=False, transform=eval_tf, download=download)
        dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
        dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
        dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, dl_test

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    es_patience: int = 20
    grad_clip: Optional[float] = None
    lambda_recon: float = 1.0
    lambda_consistency: float = 1.0
    lambda_barlow: float = 0.0
    projector_dim: Optional[int] = None
    two_views: bool = True
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
    pooling_indices: Tuple[int, ...] = (0,2)

def barlow_twins_loss(z1, z2):
    B, D = z1.shape
    z1 = (z1 - z1.mean(0)) / (z1.std(0) + 1e-9)
    z2 = (z2 - z2.mean(0)) / (z2.std(0) + 1e-9)
    c = (z1.T @ z2) / B
    on = torch.diagonal(c).add_(-1).pow_(2).sum()
    off = (c - torch.diag(torch.diagonal(c))).pow_(2).sum()
    return on + off

class InnerTrainer:
    def __init__(self, model: AutoencoderSSL, train_c: TrainConfig):
        self.model = model; self.cfg = train_c
        self.device = train_c.device; self.model.to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=train_c.lr, weight_decay=train_c.weight_decay)
        self.best_val = float("inf"); self.best_state = None; self.epochs_done = 0
    def _recon_loss(self, x, rec): return F.mse_loss(rec, x)
    def _step_unsup(self, batch, train=True):
        x = batch.to(self.device, non_blocking=True) if isinstance(batch, torch.Tensor) else batch[0].to(self.device)
        if train: self.optim.zero_grad(set_to_none=True)
        rec, z = self.model(x); loss = self._recon_loss(x, rec)
        if train:
            loss.backward()
            if self.cfg.grad_clip is not None: nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
        return float(loss.item())
    def _step_ssl(self, batch, train=True):
        x1, x2 = batch; x1 = x1.to(self.device, non_blocking=True); x2 = x2.to(self.device, non_blocking=True)
        if train: self.optim.zero_grad(set_to_none=True)
        rec1, z1 = self.model(x1); rec2, z2 = self.model(x2)
        loss_r = self._recon_loss(x1, rec1) + self._recon_loss(x2, rec2)
        loss_c = F.mse_loss(z1, z2)
        loss_b = barlow_twins_loss(z1, z2) if self.cfg.lambda_barlow > 0 else torch.tensor(0.0, device=self.device)
        loss = self.cfg.lambda_recon*loss_r + self.cfg.lambda_consistency*loss_c + self.cfg.lambda_barlow*loss_b
        if train:
            loss.backward()
            if self.cfg.grad_clip is not None: nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
        return float(loss.item())
    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.model.eval(); tot, n = 0.0, 0
        if self.cfg.two_views:
            for batch in loader:
                loss = self._step_ssl(batch, train=False); b = batch[0].size(0); tot += loss * b; n += b
        else:
            for batch in loader:
                loss = self._step_unsup(batch, train=False)
                b = batch.size(0) if isinstance(batch, torch.Tensor) else batch[0].size(0); tot += loss * b; n += b
        return tot / max(n,1)
    def fit(self, dl_train, dl_val, max_epochs=200):
        es, self.best_val, self.best_state = 0, float("inf"), None
        for _ in range(max_epochs):
            self.model.train()
            if self.cfg.two_views:
                for batch in dl_train: self._step_ssl(batch, train=True)
            else:
                for batch in dl_train: self._step_unsup(batch, train=True)
            val = self._eval_epoch(dl_val); self.epochs_done += 1
            if val + 1e-12 < self.best_val:
                self.best_val = val; self.best_state = {"model": {k: v.detach().cpu().clone() for k,v in self.model.state_dict().items()}}; es = 0
            else:
                es += 1
            if es >= self.cfg.es_patience: break
        if self.best_state is not None: self.model.load_state_dict(self.best_state["model"])
        return self.best_val

def snapshot(model: AutoencoderSSL):
    return {"state": {k: v.detach().cpu().clone() for k,v in model.state_dict().items()},
            "widths": model.widths.copy(), "pools": list(model._pools_here)}

def restore(model: AutoencoderSSL, snap):
    curr_w, target_w = model.widths, snap["widths"]
    if curr_w != target_w:
        new_model = AutoencoderSSL(in_ch=model.in_ch, widths=target_w,
                                   pooling_indices=[i for i,f in enumerate(snap["pools"]) if f],
                                   bias=model.bias, proj_dim=model.projector.out_features)
        new_model.load_state_dict(snap["state"])
        model.encoder = new_model.encoder; model.decoder = new_model.decoder
        model._pools_here = new_model._pools_here; model.projector = new_model.projector
        model.recon_head = new_model.recon_head; model.gap = new_model.gap
    else:
        model.load_state_dict(snap["state"])

def can_widen(model: AutoencoderSSL, ex_k: int, scfg: 'SearchConfig'):
    if ex_k <= 0: return False
    projected = model.total_neurons() + ex_k*(len(model.encoder)+len(model.decoder)) + ex_k
    if projected > scfg.max_neurons: return False
    if any(w + ex_k > scfg.max_width for w in model.widths): return False
    return True

def can_deepen(model: AutoencoderSSL, scfg: 'SearchConfig'):
    if len(model.encoder)+1 > scfg.max_depth: return False
    projected = model.total_neurons() + 2*model.encoder[-1].bn.num_features
    return projected <= scfg.max_neurons

def _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs): return InnerTrainer(model, tcfg).fit(dl_train, dl_val, max_epochs=max_epochs)

def ae_ssl_width_to_depth(model, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(model); best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    width_fails = 0
    while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg):
        pre = snapshot(model); model.widen_all(scfg.ex_k)
        v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(model)
            depth_fails = 0
            while depth_fails < scfg.patience_depth and can_deepen(model, scfg):
                pre2 = snapshot(model); model.append_depth()
                v2 = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(model)
                else: depth_fails += 1; restore(model, pre2)
        else:
            width_fails += 1; restore(model, pre)
    restore(model, best_snap); return model

def ae_ssl_depth_to_width(model, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(model); best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    depth_fails = 0
    while depth_fails < scfg.patience_depth and can_deepen(model, scfg):
        pre = snapshot(model); model.append_depth()
        v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(model)
            width_fails = 0
            while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg):
                pre2 = snapshot(model); model.widen_all(scfg.ex_k)
                v2 = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(model)
                else: width_fails += 1; restore(model, pre2)
        else:
            depth_fails += 1; restore(model, pre)
    restore(model, best_snap); return model

def ae_ssl_alt_depth_first(model, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(model); best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(model, scfg) and ok(total):
            pre = snapshot(model); model.append_depth()
            v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(model); improved = True
            else: depth_fails += 1; restore(model, pre)
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(model); model.widen_all(scfg.ex_k)
            v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(model); improved = True
            else: width_fails += 1; restore(model, pre)
    restore(model, best_snap); return model

def ae_ssl_alt_width_first(model, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(model); best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(model); model.widen_all(scfg.ex_k)
            v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(model); improved = True
            else: width_fails += 1; restore(model, pre)
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(model, scfg) and ok(total):
            pre = snapshot(model); model.append_depth()
            v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(model); improved = True
            else: depth_fails += 1; restore(model, pre)
    restore(model, best_snap); return model

def ae_ssl_depth_only(model, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(model); best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_depth and can_deepen(model, scfg):
        pre = snapshot(model); model.append_depth()
        v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(model)
        else: fails += 1; restore(model, pre)
    restore(model, best_snap); return model

def ae_ssl_width_only(model, dl_train, dl_val, tcfg, scfg, max_epochs=200):
    best_snap = snapshot(model); best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg):
        pre = snapshot(model); model.widen_all(scfg.ex_k)
        v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(model)
        else: fails += 1; restore(model, pre)
    restore(model, best_snap); return model
