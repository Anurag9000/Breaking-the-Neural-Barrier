
import math
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms

# ===========================
# Positional / timestep embed
# ===========================

def sinusoidal_embedding(timesteps: torch.Tensor, dim: int) -> torch.Tensor:
    """
    timesteps: (B,) int or float in [0, T-1]
    returns (B, dim)
    """
    device = timesteps.device
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(0, half, device=device) / max(half-1, 1))
    args = timesteps.float().unsqueeze(1) * freqs.unsqueeze(0)
    emb = torch.cat([torch.sin(args), torch.cos(args)], dim=1)
    if dim % 2 == 1:
        emb = F.pad(emb, (0,1))
    return emb

# ===========================
# UNet blocks
# ===========================

class ConvBNReLU(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=None, bias=True):
        super().__init__()
        if p is None: p = k // 2
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p, bias=bias)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU(inplace=True)
    def forward(self, x): return self.act(self.bn(self.conv(x)))

class FiLM(nn.Module):
    """Applies scale/shift conditioning from a vector (e.g., timestep embedding)."""
    def __init__(self, ch: int, emb_dim: int):
        super().__init__()
        self.to_scale = nn.Linear(emb_dim, ch, bias=True)
        self.to_shift = nn.Linear(emb_dim, ch, bias=True)
    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        # x: (B, C, H, W), e: (B, E)
        scale = self.to_scale(e).unsqueeze(-1).unsqueeze(-1)
        shift = self.to_shift(e).unsqueeze(-1).unsqueeze(-1)
        return x * (1 + scale) + shift

class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, emb_dim: int, bias=True):
        super().__init__()
        self.conv1 = ConvBNReLU(in_ch, out_ch, bias=bias)
        self.film = FiLM(out_ch, emb_dim)
        self.conv2 = ConvBNReLU(out_ch, out_ch, bias=bias)
        self.skip = nn.Conv2d(in_ch, out_ch, 1, bias=bias) if in_ch != out_ch else nn.Identity()
    def forward(self, x: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        h = self.conv1(x)
        h = self.film(h, e)
        h = self.conv2(h)
        return h + self.skip(x)

class Downsample(nn.Module):
    def __init__(self, ch: int): super().__init__(); self.op = nn.Conv2d(ch, ch, 3, 2, 1)
    def forward(self, x): return self.op(x)

class Upsample(nn.Module):
    def __init__(self, ch: int): super().__init__(); self.op = nn.ConvTranspose2d(ch, ch, 4, 2, 1)
    def forward(self, x): return self.op(x)

# ===========================
# Adaptive UNet (single-model)
# ===========================

class AdaptiveUNet(nn.Module):
    """
    UNet with depth/width mutability.
    - widths: channels per encoder block (no. of blocks = depth).
    - pooling_indices: after which encoder block (0-based) to downsample. Decoding mirrors those with Upsample.
    - Single noise-prediction head for DDPM-style training.
    - Timestep conditioning via FiLM in each ResBlock.
    """
    def __init__(self, in_ch: int, widths: List[int], pooling_indices: List[int], emb_dim: int = 256, bias: bool = True):
        super().__init__()
        assert len(widths) >= 1
        self.in_ch = in_ch
        self.bias = bias
        self.pooling_indices = sorted(set(pooling_indices))
        self.emb_dim = emb_dim

        self.time_mlp = nn.Sequential(
            nn.Linear(emb_dim, emb_dim), nn.SiLU(), nn.Linear(emb_dim, emb_dim)
        )

        # Encoder
        enc_blocks = nn.ModuleList()
        downs = nn.ModuleList()
        c = in_ch
        self._pools_here = []
        for i, w in enumerate(widths):
            enc_blocks.append(ResBlock(c, w, emb_dim, bias=bias))
            c = w
            do_pool = (i in self.pooling_indices)
            self._pools_here.append(do_pool)
            if do_pool:
                downs.append(Downsample(c))
            else:
                downs.append(nn.Identity())
        self.encoder = enc_blocks
        self.downs = downs

        # Bottleneck
        self.mid = ResBlock(c, c, emb_dim, bias=bias)

        # Decoder (mirror)
        dec_blocks = nn.ModuleList()
        ups = nn.ModuleList()
        for i in range(len(widths)-1, -1, -1):
            w_out = widths[i-1] if i-1 >= 0 else in_ch
            dec_blocks.append(ResBlock(c, w_out, emb_dim, bias=bias))
            c = w_out
            if self._pools_here[i]:
                ups.append(Upsample(c))
            else:
                ups.append(nn.Identity())
        self.decoder = dec_blocks
        self.ups = ups

        # Head: predict noise (same channels as input)
        self.head = nn.Conv2d(in_ch, in_ch, 3, 1, 1)

    @property
    def widths(self) -> List[int]:
        return [blk.conv1.bn.num_features for blk in self.encoder]

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # t: (B,) timesteps
        e = self.time_mlp(sinusoidal_embedding(t, self.emb_dim))

        feats = []
        h = x
        # Encoder path
        for blk, down, pooled in zip(self.encoder, self.downs, self._pools_here):
            h = blk(h, e)
            feats.append(h)
            if pooled:
                h = down(h)

        # Middle
        h = self.mid(h, e)

        # Decoder path (simple, no skip concat to keep single-model mutation simple)
        for blk, up in zip(self.decoder, self.ups):
            h = blk(h, e)
            h = up(h)

        # Output
        out = self.head(h)
        return out

    # ------------- mutations -------------
    def append_depth(self):
        last_c = self.encoder[-1].conv1.bn.num_features
        new_enc = ResBlock(last_c, last_c, self.emb_dim, bias=self.bias)
        self.encoder.append(new_enc)
        self._pools_here.append(False)
        self.downs.append(nn.Identity())

        # decoder mirrors: add block at front that maps last_c -> ? Keep symmetric with input channels of previous first dec block
        # Current decoder[0] expects in_ch == current feature ch. Append at *front* another block mapping last_c->last_c
        new_dec = ResBlock(last_c, last_c, self.emb_dim, bias=self.bias)
        self.decoder.insert(0, new_dec)
        self.ups.insert(0, nn.Identity())

    def widen_all(self, ex_k: int):
        if ex_k <= 0: return
        # Encoder widen
        prev = self.in_ch
        for blk in self.encoder:
            old_out = blk.conv1.bn.num_features
            new_out = old_out + ex_k
            _resize_resblock_(blk, prev, new_out, self.emb_dim)
            prev = new_out
        # Mid
        _resize_resblock_(self.mid, prev, prev, self.emb_dim)
        # Decoder: rebuild with transplant using updated widths
        enc_widths = [b.conv1.bn.num_features for b in self.encoder]
        self._rebuild_decoder(enc_widths)

        # Head (in_ch unchanged)
        _resize_conv2d_(self.head, in_ch=self.in_ch, out_ch=self.in_ch)

    def _rebuild_decoder(self, enc_widths: List[int]):
        old_decoder = self.decoder
        old_ups = self.ups
        dec = nn.ModuleList()
        ups = nn.ModuleList()
        c = enc_widths[-1]
        for i in range(len(enc_widths)-1, -1, -1):
            w_out = enc_widths[i-1] if i-1 >= 0 else self.in_ch
            blk = ResBlock(c, w_out, self.emb_dim, bias=self.bias)
            dec.append(blk); c = w_out
            ups.append(Upsample(c) if self._pools_here[i] else nn.Identity())

        # transplant overlap weights
        for nb, ob in zip(dec, old_decoder):
            _overlap_resblock_(nb, ob)
        self.decoder = dec
        self.ups = ups

    def total_neurons(self) -> int:
        enc = sum([b.conv1.bn.num_features for b in self.encoder])
        dec = sum([b.conv1.bn.num_features for b in self.decoder])
        mid = self.mid.conv1.bn.num_features
        return enc + dec + mid

# ===========================
# Resize utils
# ===========================

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

def _resize_resblock_(blk: ResBlock, in_ch_new: int, out_ch_new: int, emb_dim: int):
    # Save old
    old = blk
    # Conv1
    _resize_conv2d_(blk.conv1.conv, in_ch=in_ch_new, out_ch=out_ch_new)
    _resize_bn2d_(blk.conv1.bn, out_ch_new)
    # FiLM stays same emb_dim
    # Conv2
    _resize_conv2d_(blk.conv2.conv, in_ch=out_ch_new, out_ch=out_ch_new)
    _resize_bn2d_(blk.conv2.bn, out_ch_new)
    # Skip
    if isinstance(blk.skip, nn.Conv2d):
        _resize_conv2d_(blk.skip, in_ch=in_ch_new, out_ch=out_ch_new)

def _overlap_resblock_(dst: ResBlock, src: ResBlock):
    _overlap_copy_(dst.conv1.conv.weight.data, src.conv1.conv.weight.data)
    _overlap_copy_(dst.conv1.bn.weight.data, src.conv1.bn.weight.data)
    _overlap_copy_(dst.conv1.bn.bias.data, src.conv1.bn.bias.data)
    _overlap_copy_(dst.conv2.conv.weight.data, src.conv2.conv.weight.data)
    _overlap_copy_(dst.conv2.bn.weight.data, src.conv2.bn.weight.data)
    _overlap_copy_(dst.conv2.bn.bias.data, src.conv2.bn.bias.data)
    if isinstance(dst.skip, nn.Conv2d) and isinstance(src.skip, nn.Conv2d):
        _overlap_copy_(dst.skip.weight.data, src.skip.weight.data)
        if dst.skip.bias is not None and src.skip.bias is not None:
            _overlap_copy_(dst.skip.bias.data, src.skip.bias.data)

# ===========================
# Data
# ===========================

def make_cifar10_loaders_diff(data_root: str, batch_size: int, num_workers: int = 4, val_split: float = 0.1, download: bool = True, seed: int = 0):
    # Normalize to [-1, 1] for diffusion
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5))])
    ds_full = datasets.CIFAR10(root=data_root, train=True, transform=tf, download=download)
    n_val = int(len(ds_full) * val_split); n_train = len(ds_full) - n_val
    g = torch.Generator().manual_seed(seed)
    ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=g)
    ds_test = datasets.CIFAR10(root=data_root, train=False, transform=tf, download=download)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    dl_test = DataLoader(ds_test, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val, dl_test

# ===========================
# DDPM noise schedule + trainer
# ===========================

@dataclass
class DiffConfig:
    timesteps: int = 1000
    beta_start: float = 1e-4
    beta_end: float = 0.02

@dataclass
class TrainConfig:
    lr: float = 2e-4
    weight_decay: float = 0.0
    es_patience: int = 20
    grad_clip: Optional[float] = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

@dataclass
class SearchConfig:
    delta: float = 1e-3
    patience_width: int = 3
    patience_depth: int = 3
    ex_k: int = 16
    max_neurons: int = 2_000_000
    max_depth: int = 32
    max_width: int = 1024
    max_total_epochs: Optional[int] = None
    pooling_indices: Tuple[int, ...] = (0, 2)

class DiffusionHelper:
    def __init__(self, cfg: DiffConfig, device: str):
        self.T = cfg.timesteps
        betas = torch.linspace(cfg.beta_start, cfg.beta_end, self.T, dtype=torch.float32)
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register(betas, alphas, alphas_cumprod, device)

    def register(self, betas, alphas, alphas_cumprod, device):
        self.betas = betas.to(device)
        self.alphas = alphas.to(device)
        self.alphas_cumprod = alphas_cumprod.to(device)

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_alpha_bar = self.alphas_cumprod[t].sqrt().view(-1, 1, 1, 1)
        sqrt_one_minus = (1.0 - self.alphas_cumprod[t]).sqrt().view(-1, 1, 1, 1)
        return sqrt_alpha_bar * x0 + sqrt_one_minus * noise, noise

class InnerTrainer:
    def __init__(self, net: AdaptiveUNet, tcfg: TrainConfig, dcfg: DiffConfig):
        self.net = net; self.tcfg = tcfg
        self.device = tcfg.device; self.net.to(self.device)
        self.optim = torch.optim.AdamW(self.net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        self.best_val = float("inf"); self.best_state = None; self.epochs_done = 0
        self.diff = DiffusionHelper(dcfg, self.device)

    def _step_batch(self, x: torch.Tensor, train=True) -> float:
        B = x.size(0)
        t = torch.randint(0, self.diff.T, (B,), device=self.device, dtype=torch.long)
        x_noisy, eps = self.diff.q_sample(x, t)
        if train: self.optim.zero_grad(set_to_none=True)
        eps_pred = self.net(x_noisy, t)
        loss = F.mse_loss(eps_pred, eps)
        if train:
            loss.backward()
            if self.tcfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(self.net.parameters(), self.tcfg.grad_clip)
            self.optim.step()
        return float(loss.item())

    @torch.no_grad()
    def _eval_epoch(self, loader):
        self.net.eval(); tot, n = 0.0, 0
        for x, _ in loader:
            x = x.to(self.device, non_blocking=True)
            l = self._step_batch(x, train=False); b = x.size(0)
            tot += l * b; n += b
        return tot / max(n,1)

    def fit(self, dl_train, dl_val, max_epochs=50):
        es, self.best_val, self.best_state = 0, float("inf"), None
        for _ in range(max_epochs):
            self.net.train()
            for x, _ in dl_train:
                x = x.to(self.device, non_blocking=True)
                self._step_batch(x, train=True)
            val = self._eval_epoch(dl_val); self.epochs_done += 1
            if val + 1e-12 < self.best_val:
                self.best_val = val; self.best_state = {"model": {k: v.detach().cpu().clone() for k,v in self.net.state_dict().items()}}; es = 0
            else:
                es += 1
            if es >= self.tcfg.es_patience: break
        if self.best_state is not None: self.net.load_state_dict(self.best_state["model"])
        return self.best_val

# ===========================
# Snapshot & guards
# ===========================

def snapshot(net: AdaptiveUNet):
    return {"state": {k: v.detach().cpu().clone() for k,v in net.state_dict().items()},
            "widths": net.widths.copy(), "pools": list(net._pools_here)}

def restore(net: AdaptiveUNet, snap):
    curr, target = net.widths, snap["widths"]
    if curr != target:
        new = AdaptiveUNet(in_ch=net.in_ch, widths=target, pooling_indices=[i for i,f in enumerate(snap["pools"]) if f], emb_dim=net.emb_dim, bias=net.bias)
        new.load_state_dict(snap["state"])
        net.encoder = new.encoder; net.downs = new.downs; net._pools_here = new._pools_here
        net.mid = new.mid; net.decoder = new.decoder; net.ups = new.ups; net.head = new.head; net.time_mlp = new.time_mlp
    else:
        net.load_state_dict(snap["state"])

def can_widen(net: AdaptiveUNet, ex_k: int, scfg: SearchConfig) -> bool:
    if ex_k <= 0: return False
    projected = net.total_neurons() + ex_k*(len(net.encoder) + len(net.decoder) + 1)  # rough
    if projected > scfg.max_neurons: return False
    if any(w + ex_k > scfg.max_width for w in net.widths): return False
    return True

def can_deepen(net: AdaptiveUNet, scfg: SearchConfig) -> bool:
    if len(net.encoder) + 1 > scfg.max_depth: return False
    projected = net.total_neurons() + 2 * net.encoder[-1].conv1.bn.num_features
    return projected <= scfg.max_neurons

# ===========================
# Six ADP searchers
# ===========================

def _train_eval_val(net: AdaptiveUNet, dl_train, dl_val, tcfg: TrainConfig, dcfg: DiffConfig, max_epochs: int) -> float:
    return InnerTrainer(net, tcfg, dcfg).fit(dl_train, dl_val, max_epochs=max_epochs)

def diff_width_to_depth(net, dl_train, dl_val, tcfg, scfg, dcfg, max_epochs=50):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
    width_fails = 0
    while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(net)
            depth_fails = 0
            while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
                pre2 = snapshot(net); net.append_depth()
                v2 = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(net)
                else: depth_fails += 1; restore(net, pre2)
        else:
            width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def diff_depth_to_width(net, dl_train, dl_val, tcfg, scfg, dcfg, max_epochs=50):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
    depth_fails = 0
    while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(net)
            width_fails = 0
            while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
                pre2 = snapshot(net); net.widen_all(scfg.ex_k)
                v2 = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(net)
                else: width_fails += 1; restore(net, pre2)
        else:
            depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def diff_alt_depth_first(net, dl_train, dl_val, tcfg, scfg, dcfg, max_epochs=50):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def diff_alt_width_first(net, dl_train, dl_val, tcfg, scfg, dcfg, max_epochs=50):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def diff_depth_only(net, dl_train, dl_val, tcfg, scfg, dcfg, max_epochs=50):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs); fails = 0
    while fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def diff_width_only(net, dl_train, dl_val, tcfg, scfg, dcfg, max_epochs=50):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs); fails = 0
    while fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v = _train_eval_val(net, dl_train, dl_val, tcfg, dcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net
