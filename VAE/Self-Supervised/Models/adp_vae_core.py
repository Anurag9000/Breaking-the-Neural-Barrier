
import math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ======================================
# Data (torchvision)
# ======================================
try:
    from torchvision import datasets, transforms
    TV_OK = True
except Exception as e:
    TV_OK = False
    TV_ERR = e

# ======================================
# Utility blocks
# ======================================

def conv_block(in_ch, out_ch, downsample: bool, dropout: float):
    stride = 2 if downsample else 1
    pad = 1
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=stride, padding=pad, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
        nn.Dropout2d(dropout)
    )

def deconv_block(in_ch, out_ch, upsample: bool, dropout: float):
    if upsample:
        block = nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
    else:
        block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
    return block

# Simple overlap copy for convs/linears when widening
def _safe_overlap_module(dst: nn.Module, src: nn.Module):
    sd_dst = dst.state_dict()
    sd_src = src.state_dict()
    common = {k: v for k, v in sd_src.items() if k in sd_dst and sd_dst[k].shape == v.shape}
    sd_dst.update(common)
    dst.load_state_dict(sd_dst)

def _safe_overlap_linear(dst: nn.Linear, src: nn.Linear):
    with torch.no_grad():
        h = min(dst.weight.shape[0], src.weight.shape[0])
        w = min(dst.weight.shape[1], src.weight.shape[1])
        dst.weight[:h, :w].copy_(src.weight[:h, :w])
        if dst.bias is not None and src.bias is not None:
            b = min(dst.bias.shape[0], src.bias.shape[0])
            dst.bias[:b].copy_(src.bias[:b])

# ======================================
# VAE
# ======================================

class ConvVAE(nn.Module):
    def __init__(self, in_ch: int, img_size: int, channels: List[int], z_dim: int,
                 downsample_pattern: List[bool],
                 dropout: float = 0.0, recon: str = "bce"):
        super().__init__()
        assert len(channels) >= 1
        assert len(downsample_pattern) == len(channels), "downsample_pattern must match channels length"
        self.in_ch = in_ch
        self.img_size = img_size
        self.channels = list(channels)
        self.z_dim = z_dim
        self.downs = list(downsample_pattern)
        self.dropout = dropout
        self.recon = recon

        # Encoder
        enc = nn.ModuleList()
        c = in_ch
        H = img_size
        for i, (h, ds) in enumerate(zip(self.channels, self.downs)):
            enc.append(conv_block(c, h, downsample=ds, dropout=dropout))
            c = h
            if ds: H //= 2
        self.enc = enc
        self.enc_feat_h = H
        self.enc_feat_c = c
        flat_dim = c * H * H
        self.mu = nn.Linear(flat_dim, z_dim)
        self.logvar = nn.Linear(flat_dim, z_dim)

        # Decoder
        dec = nn.ModuleList()
        c = self.channels[-1]
        H = self.enc_feat_h
        # project z -> feature map
        self.dec_proj = nn.Linear(z_dim, c * H * H)
        # mirror blocks in reverse (upsample where encoder downsampled)
        for i in reversed(range(len(self.channels))):
            out_c = self.channels[i-1] if i > 0 else in_ch
            up = self.downs[i]
            dec.append(deconv_block(c, out_c, upsample=up, dropout=dropout))
            if up: H *= 2
            c = out_c
        self.dec = dec
        # output head
        self.head = nn.Conv2d(in_ch, in_ch, kernel_size=1)

    # --------- forward / sample ---------
    def encode(self, x):
        h = x
        for blk in self.enc:
            h = blk(h)
        h = h.view(h.size(0), -1)
        mu = self.mu(h)
        logvar = self.logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_proj(z)
        h = h.view(z.size(0), self.channels[-1], self.enc_feat_h, self.enc_feat_h)
        for blk in self.dec:
            h = blk(h)
        x_hat_logits = self.head(h)
        return x_hat_logits

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        x_hat_logits = self.decode(z)
        return x_hat_logits, mu, logvar

    # --------- capacity / structure helpers ---------
    @property
    def widths(self) -> List[int]:
        return self.channels

    def total_neurons(self) -> int:
        return sum(self.channels) * (self.enc_feat_h ** 2) + self.mu.in_features + self.mu.out_features + self.logvar.out_features

    def append_depth(self):
        # add one block at encoder tail (no downsample by default), and mirrored decoder block
        last = self.channels[-1]
        self.channels.append(last)
        self.downs.append(False)
        self._rebuild()

    def widen_all(self, ex_k: int):
        if ex_k <= 0: return
        self.channels = [c + ex_k for c in self.channels]
        self._rebuild()

    def _rebuild(self):
        # rebuild encoder/decoder while preserving overlap
        in_ch = self.in_ch; H = self.img_size
        new_enc = nn.ModuleList(); c = in_ch
        for h, ds in zip(self.channels, self.downs):
            nb = conv_block(c, h, downsample=ds, dropout=self.dropout)
            if len(new_enc) < len(self.enc):
                _safe_overlap_module(nb, self.enc[len(new_enc)])
            new_enc.append(nb)
            c = h; H = H//2 if ds else H
        self.enc = new_enc
        self.enc_feat_h = H
        self.enc_feat_c = c
        flat_dim = c * H * H
        new_mu = nn.Linear(flat_dim, self.z_dim)
        new_logvar = nn.Linear(flat_dim, self.z_dim)
        if hasattr(self, "mu"): _safe_overlap_linear(new_mu, self.mu)
        if hasattr(self, "logvar"): _safe_overlap_linear(new_logvar, self.logvar)
        self.mu, self.logvar = new_mu, new_logvar

        # decoder
        new_dec = nn.ModuleList(); c = self.channels[-1]; H = self.enc_feat_h
        new_dec_proj = nn.Linear(self.z_dim, c * H * H)
        if hasattr(self, "dec_proj"): _safe_overlap_linear(new_dec_proj, self.dec_proj)
        for i in reversed(range(len(self.channels))):
            out_c = self.channels[i-1] if i > 0 else self.in_ch
            up = self.downs[i]
            nb = deconv_block(c, out_c, upsample=up, dropout=self.dropout)
            if len(new_dec) < len(self.dec):
                _safe_overlap_module(nb, self.dec[len(new_dec)])
            new_dec.append(nb)
            if up: H *= 2
            c = out_c
        self.dec = new_dec
        self.dec_proj = new_dec_proj
        new_head = nn.Conv2d(self.in_ch, self.in_ch, kernel_size=1)
        if hasattr(self, "head"):
            _safe_overlap_module(new_head, self.head)
        self.head = new_head

# ======================================
# Data loaders & augmentations
# ======================================

def make_loaders(dataset: str, data_root: str, batch_size: int, num_workers: int = 0):
    if not TV_OK:
        raise RuntimeError(f"torchvision is required. Import error: {TV_ERR}")
    ds = dataset.lower()
    if ds == "mnist":
        in_ch, img_size = 1, 28
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root=data_root, train=True, transform=tfm, download=True)
        test  = datasets.MNIST(root=data_root, train=False, transform=tfm, download=True)
    elif ds == "fashionmnist":
        in_ch, img_size = 1, 28
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST(root=data_root, train=True, transform=tfm, download=True)
        test  = datasets.FashionMNIST(root=data_root, train=False, transform=tfm, download=True)
    elif ds == "cifar10":
        in_ch, img_size = 3, 32
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10(root=data_root, train=True, transform=tfm, download=True)
        test  = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=True)
    else:
        raise ValueError(f"Unknown dataset {dataset}. Choose mnist|fashionmnist|cifar10")
    dl_train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dl_val   = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return dl_train, dl_val, in_ch, img_size

def add_noise(x, std: float):
    if std <= 0: return x
    return torch.clamp(x + std * torch.randn_like(x), 0.0, 1.0)

def cutout(x, size=8, p=0.0):
    if p <= 0: return x
    B, C, H, W = x.shape
    for b in range(B):
        if random.random() < p:
            y = random.randint(0, H-1); x0 = random.randint(0, W-1)
            y1 = max(0, y - size//2); y2 = min(H, y + size//2)
            x1 = max(0, x0 - size//2); x2 = min(W, x0 + size//2)
            x[b, :, y1:y2, x1:x2] = 0.0
    return x

# ======================================
# Losses (ELBO + extras)
# ======================================

def recon_loss(x_logits, x, kind: str):
    if kind == "bce":
        return F.binary_cross_entropy_with_logits(x_logits, x, reduction="mean")
    elif kind == "mse":
        x_hat = torch.sigmoid(x_logits)
        return F.mse_loss(x_hat, x, reduction="mean")
    elif kind == "l1":
        x_hat = torch.sigmoid(x_logits)
        return F.l1_loss(x_hat, x, reduction="mean")
    else:
        raise ValueError(f"Unknown recon kind {kind}")

def kl_loss(mu, logvar):
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def mmd_isotropic_gaussian(z):
    # Simple MMD with RBF kernels against N(0,I) samples
    B, D = z.shape
    z_prior = torch.randn_like(z)
    def pdist(a):  # squared pairwise distances
        a2 = (a*a).sum(dim=1, keepdim=True)
        return a2 + a2.t() - 2*a @ a.t()
    def rbf(d2, sigma2):
        return torch.exp(-d2 / (2*sigma2))
    d_xx = pdist(z); d_pp = pdist(z_prior); d_xp = (z.unsqueeze(1)-z_prior.unsqueeze(0)).pow(2).sum(dim=2)
    sigmas = [0.1, 1.0, 2.0, 5.0]
    k_xx = sum(rbf(d_xx, s**2).mean() for s in sigmas) / len(sigmas)
    k_pp = sum(rbf(d_pp, s**2).mean() for s in sigmas) / len(sigmas)
    k_xp = sum(rbf(d_xp, s**2).mean() for s in sigmas) / len(sigmas)
    return k_xx + k_pp - 2*k_xp

# ======================================
# Configs
# ======================================

@dataclass
class TrainConfig:
    lr: float = 1e-3
    weight_decay: float = 0.0
    es_patience: int = 20
    grad_clip: Optional[float] = 1.0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    beta: float = 4.0
    recon: str = "bce"
    kl_warmup_epochs: int = 20
    noise_std: float = 0.1
    cutout_p: float = 0.0
    # self-supervised extras
    lambda_align: float = 0.0   # latent mean alignment between two views
    lambda_mmd: float = 0.0     # InfoVAE/β-TCVAE style MMD regularizer

@dataclass
class SearchConfig:
    delta: float = 1e-3
    patience_width: int = 2
    patience_depth: int = 2
    ex_k: int = 16
    max_neurons: int = 3_000_000
    max_depth: int = 16
    max_width: int = 1024
    max_total_epochs: Optional[int] = None
    down_every: int = 2  # downsample every N blocks initially

# ======================================
# Trainer
# ======================================

class InnerTrainer:
    def __init__(self, net: ConvVAE, tcfg: TrainConfig, z_dim: int):
        self.net = net; self.tcfg = tcfg; self.z_dim = z_dim
        self.net.to(tcfg.device)
        self.optim = torch.optim.AdamW(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        self.best = float("inf"); self.best_state = None; self.epochs_done = 0

    def _elbo_batch(self, x, epoch_idx: int, selfsup: bool = True):
        x = x.to(self.tcfg.device)
        clean = x.clone()
        x_noisy = add_noise(x, self.tcfg.noise_std)
        x_noisy = cutout(x_noisy, p=self.tcfg.cutout_p)
        x_in = x_noisy
        x_logits, mu, logvar = self.net(x_in)
        recon = recon_loss(x_logits, clean, self.tcfg.recon)
        beta_t = self.tcfg.beta * min(1.0, max(0.0, epoch_idx / max(1, self.tcfg.kl_warmup_epochs)))
        kld = kl_loss(mu, logvar)
        loss = recon + beta_t * kld

        # self-supervised latent alignment (two noisy views)
        if selfsup and self.tcfg.lambda_align > 0.0:
            x2 = add_noise(clean, self.tcfg.noise_std)
            x2 = cutout(x2, p=self.tcfg.cutout_p)
            _, mu2, _ = self.net(x2)  # share weights, compute means
            align = F.mse_loss(mu, mu2)
            loss = loss + self.tcfg.lambda_align * align

        # MMD toward N(0,I) (on posterior samples)
        if self.tcfg.lambda_mmd > 0.0:
            z = self.net.reparameterize(mu, logvar)
            mmd = mmd_isotropic_gaussian(z)
            loss = loss + self.tcfg.lambda_mmd * mmd

        return loss

    @torch.no_grad()
    def _eval_epoch(self, loader, epoch_idx: int):
        self.net.eval(); tot, n = 0.0, 0
        for x, _ in loader:
            l = self._elbo_batch(x, epoch_idx, selfsup=False)
            bsz = x.size(0)
            tot += float(l.item()) * bsz; n += bsz
        return tot / max(n, 1)

    def fit(self, dl_train, dl_val, max_epochs=50):
        es = 0
        for e in range(max_epochs):
            self.net.train()
            for x, _ in dl_train:
                self.optim.zero_grad(set_to_none=True)
                loss = self._elbo_batch(x, epoch_idx=e, selfsup=True)
                loss.backward()
                if self.tcfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.tcfg.grad_clip)
                self.optim.step()
            val = self._eval_epoch(dl_val, epoch_idx=e); self.epochs_done += 1
            if val + 1e-12 < self.best:
                self.best = val
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                es = 0
            else:
                es += 1
            if es >= self.tcfg.es_patience:
                break
        if self.best_state is not None:
            self.net.load_state_dict(self.best_state)
        return self.best

# ======================================
# Snapshot / restore / guards
# ======================================

def snapshot(net: ConvVAE):
    return {"state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            "channels": net.channels.copy(),
            "downs": net.downs.copy()}

def restore(net: ConvVAE, snap):
    if net.channels != snap["channels"] or net.downs != snap["downs"]:
        net.channels = snap["channels"].copy()
        net.downs = snap["downs"].copy()
        net._rebuild()
    net.load_state_dict(snap["state"])

def can_widen(net: ConvVAE, ex_k: int, scfg) -> bool:
    if ex_k <= 0: return False
    if any(c + ex_k > scfg.max_width for c in net.channels): return False
    projected = net.total_neurons() + ex_k * len(net.channels) * (net.enc_feat_h ** 2)
    return projected <= scfg.max_neurons

def can_deepen(net: ConvVAE, scfg) -> bool:
    if len(net.channels) + 1 > scfg.max_depth: return False
    projected = net.total_neurons() + net.channels[-1] * (net.enc_feat_h ** 2)
    return projected <= scfg.max_neurons

# ======================================
# Six ADP searchers
# ======================================

def _train_eval_val(net: ConvVAE, dl_train, dl_val, tcfg: TrainConfig, max_epochs: int) -> float:
    return InnerTrainer(net, tcfg, net.z_dim).fit(dl_train, dl_val, max_epochs=max_epochs)

def vae_width_to_depth(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
    width_fails = 0
    while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(net)
            depth_fails = 0
            while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
                pre2 = snapshot(net); net.append_depth()
                v2 = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(net)
                else: depth_fails += 1; restore(net, pre2)
        else:
            width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def vae_depth_to_width(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
    depth_fails = 0
    while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val = v; best_snap = snapshot(net)
            width_fails = 0
            while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
                pre2 = snapshot(net); net.widen_all(scfg.ex_k)
                v2 = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val = v2; best_snap = snapshot(net)
                else: width_fails += 1; restore(net, pre2)
        else:
            depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def vae_alt_depth_first(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def vae_alt_width_first(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def vae_depth_only(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def vae_width_only(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v = _train_eval_val(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val = v; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net

# ======================================
# Entry construction
# ======================================

def build_vae(in_ch, img_size, init_width, init_depth, z_dim, dropout, down_every=2, recon="bce"):
    # initial channels and downsample pattern
    channels = [init_width] * init_depth
    downs = [(i % down_every == (down_every-1)) for i in range(init_depth)]
    net = ConvVAE(in_ch, img_size, channels, z_dim, downs, dropout=dropout, recon=recon)
    return net
