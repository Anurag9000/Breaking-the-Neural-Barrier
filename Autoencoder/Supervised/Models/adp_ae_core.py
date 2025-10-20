
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# ---------------------------
# Blocks
# ---------------------------

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, bias: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

class DeconvBNReLU(nn.Module):
    """ConvTranspose2d block with BN+ReLU; mirrors ConvBNReLU but upsamples when stride=2."""
    def __init__(self, in_ch: int, out_ch: int, k: int = 3, s: int = 1, p: Optional[int] = None, outpad: int = 0, bias: bool = True):
        super().__init__()
        if p is None:
            p = k // 2
        self.deconv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, output_padding=outpad, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.deconv(x)))

# ---------------------------
# Autoencoder (supervised)
# ---------------------------

class AutoencoderCNN(nn.Module):
    """
    Supervised autoencoder:
      - Encoder: list of ConvBNReLU blocks. Optional downsampling via MaxPool2d after blocks indexed by `pooling_indices` (0-based).
      - Latent: global average pooling -> flatten -> Linear head for classification.
      - Decoder: mirrors the encoder. Uses ConvTranspose2d blocks and upsampling (via stride=2) where the encoder pooled.
      - Output: reconstruction logits via 3x3 conv (same channels as input). We predict normalized image (same preprocessing as input).
    """
    def __init__(
        self,
        in_ch: int,
        num_classes: int,
        widths: List[int],
        pooling_indices: List[int],
        bias: bool = True,
    ):
        super().__init__()
        assert len(widths) >= 1, "Need at least one encoder block"
        self.in_ch = in_ch
        self.num_classes = num_classes
        self.pooling_indices = sorted(list(set(pooling_indices)))
        self.bias = bias

        # Encoder
        enc_blocks = []
        c_in = in_ch
        for i, c_out in enumerate(widths):
            enc_blocks.append(ConvBNReLU(c_in, c_out, bias=bias))
            c_in = c_out
        self.encoder = nn.ModuleList(enc_blocks)

        # Track where we downsample to mirror in decoder
        self._pools_here = [i in self.pooling_indices for i in range(len(widths))]
        self.pool = nn.MaxPool2d(2, 2)

        # Latent + classifier head
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(widths[-1], num_classes, bias=True)

        # Decoder (mirror)
        dec_blocks = []
        # Start channel for decoder equals last encoder channel
        c_in = widths[-1]
        for i in range(len(widths) - 1, -1, -1):
            c_out = widths[i - 1] if i - 1 >= 0 else in_ch
            # If there was pooling after encoder block i, we upsample here with stride=2
            stride = 2 if self._pools_here[i] else 1
            outpad = 1 if stride == 2 else 0
            dec_blocks.append(DeconvBNReLU(c_in, c_out, k=3, s=stride, outpad=outpad, bias=bias))
            c_in = c_out
        self.decoder = nn.ModuleList(dec_blocks)

        # Final conv to map back to input channels
        self.recon_head = nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=1, padding=1, bias=True)

    @property
    def widths(self) -> List[int]:
        return [blk.bn.num_features for blk in self.encoder]

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Encoder
        h = x
        for i, blk in enumerate(self.encoder):
            h = blk(h)
            if self._pools_here[i]:
                h = self.pool(h)

        # Latent and classifier
        z = self.gap(h).flatten(1)
        logits = self.classifier(z)

        # Decoder
        h_dec = h
        for i, blk in enumerate(self.decoder):
            h_dec = blk(h_dec)
        rec = self.recon_head(h_dec)
        return rec, logits

    # -------------------
    # Mutations
    # -------------------

    def append_depth(self) -> None:
        """Append one encoder block + mirrored decoder block.
        We keep channel width equal to the last encoder channel."""
        last_c = self.encoder[-1].bn.num_features
        # Encoder add
        new_enc = ConvBNReLU(last_c, last_c, bias=self.bias)
        self.encoder.append(new_enc)
        # Pooling policy: inherit "no pool" by default for the new block
        self._pools_here.append(False)

        # Decoder: insert *at front* a mirror of this new block (since decoder iterates reverse)
        stride = 2 if self._pools_here[-1] else 1
        outpad = 1 if stride == 2 else 0
        c_in = last_c
        c_out = last_c
        new_dec = DeconvBNReLU(c_in, c_out, s=stride, outpad=outpad, bias=self.bias)
        self.decoder.insert(0, new_dec)

    def widen_all(self, ex_k: int) -> None:
        """Increase channels of every encoder block by ex_k and mirror to decoder+classifier."""
        if ex_k <= 0:
            return
        # Encoder widen
        prev_out = self.in_ch
        for i, enc in enumerate(self.encoder):
            old_out = enc.bn.num_features
            new_out = old_out + ex_k
            _resize_conv2d_(enc.conv, in_ch=prev_out, out_ch=new_out)
            _resize_bn2d_(enc.bn, new_out)
            prev_out = new_out

        # Classifier head
        old_head_in = self.classifier.in_features
        new_head_in = old_head_in + ex_k
        _resize_linear_(self.classifier, in_f=new_head_in, out_f=self.num_classes)

        # Decoder widen (mirror) by rebuild+transplant
        enc_widths = [blk.bn.num_features for blk in self.encoder]
        self._rebuild_decoder(enc_widths)

    def _rebuild_decoder(self, enc_widths: List[int]) -> None:
        old_dec = self.decoder
        new_dec = nn.ModuleList()
        c_in = enc_widths[-1]
        for i in range(len(enc_widths) - 1, -1, -1):
            c_out = enc_widths[i - 1] if i - 1 >= 0 else self.in_ch
            stride = 2 if self._pools_here[i] else 1
            outpad = 1 if stride == 2 else 0
            new_blk = DeconvBNReLU(c_in, c_out, k=3, s=stride, outpad=outpad, bias=self.bias)
            new_dec.append(new_blk)
            c_in = c_out

        for new_blk, old_blk in zip(new_dec, old_dec):
            _resize_convtranspose2d_(new_blk.deconv, in_ch=old_blk.deconv.in_channels, out_ch=old_blk.deconv.out_channels)
            _overlap_copy_(new_blk.deconv.weight.data, old_blk.deconv.weight.data)
            if new_blk.deconv.bias is not None and old_blk.deconv.bias is not None:
                _overlap_copy_(new_blk.deconv.bias.data, old_blk.deconv.bias.data)
            _resize_bn2d_(new_blk.bn, new_blk.bn.num_features)
            _overlap_copy_(new_blk.bn.weight.data, old_blk.bn.weight.data)
            _overlap_copy_(new_blk.bn.bias.data, old_blk.bn.bias.data)
            _overlap_copy_(new_blk.bn.running_mean, old_blk.bn.running_mean)
            _overlap_copy_(new_blk.bn.running_var, old_blk.bn.running_var)
        self.decoder = new_dec

    def total_neurons(self) -> int:
        enc_ch = sum([blk.bn.num_features for blk in self.encoder])
        dec_ch = sum([blk.bn.num_features for blk in self.decoder])
        return enc_ch + dec_ch + self.classifier.in_features


def _overlap_copy_(dst: torch.Tensor, src: torch.Tensor) -> None:
    dims = [min(a, b) for a, b in zip(dst.shape, src.shape)]
    slicer = tuple(slice(0, d) for d in dims)
    dst[slicer].copy_(src[slicer])


def _resize_conv2d_(conv: nn.Conv2d, in_ch: int, out_ch: int) -> None:
    old_w = conv.weight.data.clone()
    old_b = conv.bias.data.clone() if conv.bias is not None else None
    k_h, k_w = conv.kernel_size
    device = conv.weight.device
    conv.in_channels = in_ch
    conv.out_channels = out_ch
    conv.weight = nn.Parameter(torch.empty(out_ch, in_ch, k_h, k_w, device=device))
    nn.init.kaiming_normal_(conv.weight, nonlinearity='relu')
    _overlap_copy_(conv.weight.data, old_w)
    if conv.bias is not None:
        conv.bias = nn.Parameter(torch.zeros(out_ch, device=device))
        if old_b is not None:
            _overlap_copy_(conv.bias.data, old_b)

def _resize_convtranspose2d_(deconv: nn.ConvTranspose2d, in_ch: int, out_ch: int) -> None:
    old_w = deconv.weight.data.clone()
    old_b = deconv.bias.data.clone() if deconv.bias is not None else None
    k_h, k_w = deconv.kernel_size
    device = deconv.weight.device
    deconv.in_channels = in_ch
    deconv.out_channels = out_ch
    deconv.weight = nn.Parameter(torch.empty(in_ch, out_ch, k_h, k_w, device=device))  # ConvTranspose shape
    nn.init.kaiming_normal_(deconv.weight, nonlinearity='relu')
    _overlap_copy_(deconv.weight.data, old_w)
    if deconv.bias is not None:
        deconv.bias = nn.Parameter(torch.zeros(out_ch, device=device))
        if old_b is not None:
            _overlap_copy_(deconv.bias.data, old_b)

def _resize_bn2d_(bn: nn.BatchNorm2d, out_ch: int) -> None:
    device = bn.weight.device
    old_w = bn.weight.data.clone()
    old_b = bn.bias.data.clone()
    old_rm = bn.running_mean.clone()
    old_rv = bn.running_var.clone()
    bn.num_features = out_ch
    bn.weight = nn.Parameter(torch.ones(out_ch, device=device))
    bn.bias = nn.Parameter(torch.zeros(out_ch, device=device))
    bn.running_mean = torch.zeros(out_ch, device=device)
    bn.running_var = torch.ones(out_ch, device=device)
    _overlap_copy_(bn.weight.data, old_w)
    _overlap_copy_(bn.bias.data, old_b)
    _overlap_copy_(bn.running_mean, old_rm)
    _overlap_copy_(bn.running_var, old_rv)

def _resize_linear_(fc: nn.Linear, in_f: int, out_f: int) -> None:
    device = fc.weight.device
    old_w = fc.weight.data.clone()
    old_b = fc.bias.data.clone() if fc.bias is not None else None
    fc.in_features = in_f
    fc.out_features = out_f
    fc.weight = nn.Parameter(torch.empty(out_f, in_f, device=device))
    nn.init.kaiming_uniform_(fc.weight, a=math.sqrt(5))
    _overlap_copy_(fc.weight.data, old_w)
    if fc.bias is not None:
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(fc.weight)
        bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
        fc.bias = nn.Parameter(torch.empty(out_f, device=device))
        nn.init.uniform_(fc.bias, -bound, bound)
        if old_b is not None:
            _overlap_copy_(fc.bias.data, old_b)

# ---------------------------
# Data & training
# ---------------------------

def make_cifar10_loaders(
    data_root: str,
    batch_size: int,
    num_workers: int = 4,
    val_split: float = 0.1,
    download: bool = True,
    seed: int = 0,
):
    mean = (0.4914, 0.4822, 0.4465)
    std = (0.2470, 0.2435, 0.2616)
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    ds_full = datasets.CIFAR10(root=data_root, train=True, transform=tf_train, download=download)
    n_val = int(len(ds_full) * val_split)
    n_train = len(ds_full) - n_val
    g = torch.Generator().manual_seed(seed)
    ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=g)
    ds_test = datasets.CIFAR10(root=data_root, train=False, transform=tf_eval, download=download)

    from torch.utils.data import DataLoader
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, dl_test

from dataclasses import dataclass

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4
    es_patience: int = 20
    grad_clip: Optional[float] = None
    lambda_recon: float = 1.0
    lambda_cls: float = 1.0
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
    pooling_indices: tuple = (0, 2)

class InnerTrainer:
    def __init__(self, model: AutoencoderCNN, train_c: TrainConfig):
        self.model = model
        self.cfg = train_c
        self.device = train_c.device
        self.model.to(self.device)
        self.optim = torch.optim.AdamW(self.model.parameters(), lr=train_c.lr, weight_decay=train_c.weight_decay)
        self.best_val = float("inf")
        self.best_state = None
        self.epochs_done = 0

    def _step(self, batch, train: bool = True):
        x, y = batch
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        if train:
            self.optim.zero_grad(set_to_none=True)
        rec, logits = self.model(x)
        loss_recon = F.mse_loss(rec, x)
        loss_cls = F.cross_entropy(logits, y)
        loss = self.cfg.lambda_recon * loss_recon + self.cfg.lambda_cls * loss_cls
        if train:
            loss.backward()
            if self.cfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
            self.optim.step()
        return loss.item(), loss_recon.item(), loss_cls.item()

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.model.eval()
        tot, tot_r, tot_c, n = 0.0, 0.0, 0.0, 0
        for batch in loader:
            loss, lr, lc = self._step(batch, train=False)
            bs = batch[0].size(0)
            tot += loss * bs
            tot_r += lr * bs
            tot_c += lc * bs
            n += bs
        return tot / n, tot_r / n, tot_c / n

    def fit(self, dl_train, dl_val, max_epochs: int = 200):
        es_counter = 0
        self.best_val = float("inf")
        self.best_state = None
        for epoch in range(max_epochs):
            self.model.train()
            for batch in dl_train:
                self._step(batch, train=True)
            val, v_r, v_c = self._eval_epoch(dl_val)
            self.epochs_done += 1
            if val + 1e-12 < self.best_val:
                self.best_val = val
                self.best_state = {
                    "model": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
                }
                es_counter = 0
            else:
                es_counter += 1
            if es_counter >= self.cfg.es_patience:
                break
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state["model"])
        return self.best_val

def snapshot(model: AutoencoderCNN):
    return {
        "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
        "widths": model.widths.copy(),
        "pools": list(model._pools_here),
    }

def restore(model: AutoencoderCNN, snap):
    curr_w = model.widths
    target_w = snap["widths"]
    if curr_w != target_w:
        new_model = AutoencoderCNN(
            in_ch=model.in_ch,
            num_classes=model.num_classes,
            widths=target_w,
            pooling_indices=[i for i, f in enumerate(snap["pools"]) if f],
            bias=model.bias,
        )
        new_model.load_state_dict(snap["state"])
        model.encoder = new_model.encoder
        model.decoder = new_model.decoder
        model._pools_here = new_model._pools_here
        model.classifier = new_model.classifier
        model.recon_head = new_model.recon_head
        model.gap = new_model.gap
    else:
        model.load_state_dict(snap["state"])

def can_widen(model: AutoencoderCNN, ex_k: int, scfg: SearchConfig) -> bool:
    if ex_k <= 0:
        return False
    projected = model.total_neurons() + ex_k * (len(model.encoder) + len(model.decoder)) + ex_k
    if projected > scfg.max_neurons:
        return False
    if any(w + ex_k > scfg.max_width for w in model.widths):
        return False
    return True

def can_deepen(model: AutoencoderCNN, scfg: SearchConfig) -> bool:
    if len(model.encoder) + 1 > scfg.max_depth:
        return False
    projected = model.total_neurons() + 2 * model.encoder[-1].bn.num_features
    return projected <= scfg.max_neurons

def _train_eval_val(model: AutoencoderCNN, dl_train, dl_val, tcfg: TrainConfig, max_epochs: int) -> float:
    trainer = InnerTrainer(model, tcfg)
    best_val = trainer.fit(dl_train, dl_val, max_epochs=max_epochs)
    return best_val

def ae_search_width_to_depth(model, dl_train, dl_val, tcfg, scfg, max_epochs: int = 200):
    best_snap = snapshot(model)
    best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    width_fails = 0
    while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg):
        pre_snap = snapshot(model)
        model.widen_all(scfg.ex_k)
        new_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
        if new_val < best_val - scfg.delta:
            best_val = new_val
            best_snap = snapshot(model)
            depth_fails = 0
            while depth_fails < scfg.patience_depth and can_deepen(model, scfg):
                pre2 = snapshot(model)
                model.append_depth()
                d_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
                if d_val < best_val - scfg.delta:
                    best_val = d_val
                    best_snap = snapshot(model)
                else:
                    depth_fails += 1
                    restore(model, pre2)
        else:
            width_fails += 1
            restore(model, pre_snap)
    restore(model, best_snap)
    return model

def ae_search_depth_to_width(model, dl_train, dl_val, tcfg, scfg, max_epochs: int = 200):
    best_snap = snapshot(model)
    best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    depth_fails = 0
    while depth_fails < scfg.patience_depth and can_deepen(model, scfg):
        pre_snap = snapshot(model)
        model.append_depth()
        d_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
        if d_val < best_val - scfg.delta:
            best_val = d_val
            best_snap = snapshot(model)
            width_fails = 0
            while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg):
                pre2 = snapshot(model)
                model.widen_all(scfg.ex_k)
                w_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
                if w_val < best_val - scfg.delta:
                    best_val = w_val
                    best_snap = snapshot(model)
                else:
                    width_fails += 1
                    restore(model, pre2)
        else:
            depth_fails += 1
            restore(model, pre_snap)
    restore(model, best_snap)
    return model

def ae_search_alt_depth_first(model, dl_train, dl_val, tcfg, scfg, max_epochs: int = 200):
    best_snap = snapshot(model)
    best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    total_epochs = 0
    def budget_ok(ep_done: int) -> bool:
        return scfg.max_total_epochs is None or ep_done < scfg.max_total_epochs
    improved_in_cycle = True
    while improved_in_cycle and budget_ok(total_epochs):
        improved_in_cycle = False
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(model, scfg) and budget_ok(total_epochs):
            pre = snapshot(model)
            model.append_depth()
            d_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
            total_epochs += max_epochs
            if d_val < best_val - scfg.delta:
                best_val = d_val
                best_snap = snapshot(model)
                improved_in_cycle = True
            else:
                depth_fails += 1
                restore(model, pre)
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg) and budget_ok(total_epochs):
            pre = snapshot(model)
            model.widen_all(scfg.ex_k)
            w_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
            total_epochs += max_epochs
            if w_val < best_val - scfg.delta:
                best_val = w_val
                best_snap = snapshot(model)
                improved_in_cycle = True
            else:
                width_fails += 1
                restore(model, pre)
    restore(model, best_snap)
    return model

def ae_search_alt_width_first(model, dl_train, dl_val, tcfg, scfg, max_epochs: int = 200):
    best_snap = snapshot(model)
    best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    total_epochs = 0
    def budget_ok(ep_done: int) -> bool:
        return scfg.max_total_epochs is None or ep_done < scfg.max_total_epochs
    improved_in_cycle = True
    while improved_in_cycle and budget_ok(total_epochs):
        improved_in_cycle = False
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg) and budget_ok(total_epochs):
            pre = snapshot(model)
            model.widen_all(scfg.ex_k)
            w_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
            total_epochs += max_epochs
            if w_val < best_val - scfg.delta:
                best_val = w_val
                best_snap = snapshot(model)
                improved_in_cycle = True
            else:
                width_fails += 1
                restore(model, pre)
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(model, scfg) and budget_ok(total_epochs):
            pre = snapshot(model)
            model.append_depth()
            d_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
            total_epochs += max_epochs
            if d_val < best_val - scfg.delta:
                best_val = d_val
                best_snap = snapshot(model)
                improved_in_cycle = True
            else:
                depth_fails += 1
                restore(model, pre)
    restore(model, best_snap)
    return model

def ae_search_depth_only(model, dl_train, dl_val, tcfg, scfg, max_epochs: int = 200):
    best_snap = snapshot(model)
    best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    fails = 0
    while fails < scfg.patience_depth and can_deepen(model, scfg):
        pre = snapshot(model)
        model.append_depth()
        v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v
            best_snap = snapshot(model)
        else:
            fails += 1
            restore(model, pre)
    restore(model, best_snap)
    return model

def ae_search_width_only(model, dl_train, dl_val, tcfg, scfg, max_epochs: int = 200):
    best_snap = snapshot(model)
    best_val = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
    fails = 0
    while fails < scfg.patience_width and can_widen(model, scfg.ex_k, scfg):
        pre = snapshot(model)
        model.widen_all(scfg.ex_k)
        v = _train_eval_val(model, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v
            best_snap = snapshot(model)
        else:
            fails += 1
            restore(model, pre)
    restore(model, best_snap)
    return model
