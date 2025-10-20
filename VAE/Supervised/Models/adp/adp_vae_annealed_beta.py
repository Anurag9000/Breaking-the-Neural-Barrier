
import math, random
from dataclasses import dataclass
from typing import List, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ==============================
# Utility blocks
# ==============================

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
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )
    else:
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout2d(dropout)
        )

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

# ==============================
# Supervised VAE (aux head and CVAE modes)
# ==============================

class ConvVAE_Sup(nn.Module):
    def __init__(self, in_ch: int, img_size: int, channels: List[int], z_dim: int, num_classes: int,
                 downsample_pattern: List[bool],
                 dropout: float = 0.0, recon: str = "bce",
                 sup_mode: str = "aux", cvae_cond_ch: int = 16, aux_head: bool = True):
        super().__init__()
        assert len(channels) >= 1
        assert len(downsample_pattern) == len(channels), "downsample_pattern must match channels length"
        assert sup_mode in ["aux","cvae"]
        self.in_ch = in_ch
        self.img_size = img_size
        self.channels = list(channels)
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.downs = list(downsample_pattern)
        self.dropout = dropout
        self.recon = recon
        self.sup_mode = sup_mode
        self.aux_head_on = (sup_mode=="aux") or (sup_mode=="cvae" and aux_head)

        # Label embedding for CVAE
        self.cvae_cond_ch = cvae_cond_ch
        if self.sup_mode == "cvae":
            self.y_embed = nn.Embedding(num_classes, cvae_cond_ch)
            self.y_proj_enc = nn.Conv2d(cvae_cond_ch, cvae_cond_ch, kernel_size=1)
            self.y_proj_dec = nn.Conv2d(cvae_cond_ch, cvae_cond_ch, kernel_size=1)

        # -------- Encoder --------
        enc = nn.ModuleList()
        c = in_ch + (cvae_cond_ch if self.sup_mode=="cvae" else 0)
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

        # -------- Decoder --------
        dec = nn.ModuleList()
        c = self.channels[-1]
        H = self.enc_feat_h
        self.dec_proj = nn.Linear(z_dim + (0 if self.sup_mode=="aux" else num_classes), c * H * H)
        for i in reversed(range(len(self.channels))):
            out_c = self.channels[i-1] if i > 0 else in_ch
            up = self.downs[i]
            inp_c = c + (self.cvae_cond_ch if (self.sup_mode=="cvae" and i==len(self.channels)-1) else 0)
            dec.append(deconv_block(inp_c, out_c, upsample=up, dropout=dropout))
            if up: H *= 2
            c = out_c
        self.dec = dec
        self.head = nn.Conv2d(in_ch, in_ch, kernel_size=1)

        # -------- Classifier (aux) --------
        if self.aux_head_on:
            self.clf = nn.Sequential(
                nn.Linear(z_dim, max(64, z_dim)),
                nn.ReLU(inplace=True),
                nn.Dropout(0.1),
                nn.Linear(max(64, z_dim), num_classes)
            )

    # ----- helpers for CVAE conditioning -----
    def _label_map(self, y, H, W):
        emb = self.y_embed(y)  # (B, Cc)
        m = emb.view(emb.size(0), self.cvae_cond_ch, 1, 1).expand(-1, -1, H, W)
        return m

    # -------- forward pieces --------
    def encode(self, x, y=None):
        if self.sup_mode == "cvae":
            ymap = self._label_map(y, x.size(2), x.size(3))
            ymap = self.y_proj_enc(ymap)
            h = torch.cat([x, ymap], dim=1)
        else:
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

    def decode(self, z, y=None):
        if self.sup_mode == "cvae":
            y_onehot = F.one_hot(y, num_classes=self.num_classes).float()
            z_in = torch.cat([z, y_onehot], dim=1)
        else:
            z_in = z
        h = self.dec_proj(z_in)
        h = h.view(z.size(0), self.channels[-1], self.enc_feat_h, self.enc_feat_h)
        if self.sup_mode == "cvae":
            ymap = self._label_map(y, h.size(2), h.size(3))
            ymap = self.y_proj_dec(ymap)
            h = torch.cat([h, ymap], dim=1)
        for blk in self.dec:
            h = blk(h)
        x_hat_logits = self.head(h)
        return x_hat_logits

    def forward(self, x, y):
        mu, logvar = self.encode(x, y if self.sup_mode=="cvae" else None)
        z = self.reparameterize(mu, logvar)
        x_hat_logits = self.decode(z, y if self.sup_mode=="cvae" else None)
        logits = self.clf(z) if self.aux_head_on else None
        return x_hat_logits, mu, logvar, logits

    # -------- capacity / ADP structure --------
    @property
    def widths(self) -> List[int]:
        return self.channels

    def total_neurons(self) -> int:
        return sum(self.channels) * (self.enc_feat_h ** 2) + self.mu.in_features + self.mu.out_features + self.logvar.out_features

    def append_depth(self):
        last = self.channels[-1]
        self.channels.append(last)
        self.downs.append(False)
        self._rebuild()

    def widen_all(self, ex_k: int):
        if ex_k <= 0: return
        self.channels = [c + ex_k for c in self.channels]
        self._rebuild()

    def _rebuild(self):
        in_ch = self.in_ch + (self.cvae_cond_ch if self.sup_mode=="cvae" else 0)
        H = self.img_size
        new_enc = nn.ModuleList(); c = in_ch
        for idx, (h, ds) in enumerate(zip(self.channels, self.downs)):
            nb = conv_block(c, h, downsample=ds, dropout=self.dropout)
            if hasattr(self, "enc") and idx < len(self.enc):
                _safe_overlap_module(nb, self.enc[idx])
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

        # Decoder
        new_dec = nn.ModuleList(); c = self.channels[-1]; H = self.enc_feat_h
        dec_in_z = self.z_dim + (0 if self.sup_mode=="aux" else self.num_classes)
        new_dec_proj = nn.Linear(dec_in_z, c * H * H)
        if hasattr(self, "dec_proj"): _safe_overlap_linear(new_dec_proj, self.dec_proj)
        for i in reversed(range(len(self.channels))):
            out_c = self.channels[i-1] if i > 0 else self.in_ch
            up = self.downs[i]
            inp_c = c + (self.cvae_cond_ch if (self.sup_mode=="cvae" and i==len(self.channels)-1) else 0)
            nb = deconv_block(inp_c, out_c, upsample=up, dropout=self.dropout)
            if hasattr(self, "dec") and i < len(self.channels) and len(new_dec) < len(self.dec):
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

# ==============================
# Losses
# ==============================

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

# ==============================
# Configs
# ==============================

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
    lambda_sup: float = 1.0
    sup_mode: str = "aux"
    aux_head: bool = True
    min_acc_drop: float = 0.0

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
    down_every: int = 2

# ==============================
# Trainer
# ==============================

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

class InnerTrainer:
    def __init__(self, net: ConvVAE_Sup, tcfg: TrainConfig, num_classes: int):
        self.net = net; self.tcfg = tcfg; self.num_classes = num_classes
        self.net.to(tcfg.device)
        self.optim = torch.optim.AdamW(net.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
        self.best = float("inf"); self.best_state = None; self.epochs_done = 0
        self.best_acc = 0.0

    def _sup_batch(self, x, y, epoch_idx: int, eval_only=False):
        x = x.to(self.tcfg.device); y = y.to(self.tcfg.device)
        clean = x.clone()
        x_noisy = add_noise(x, self.tcfg.noise_std if not eval_only else 0.0)
        x_noisy = cutout(x_noisy, p=self.tcfg.cutout_p if not eval_only else 0.0)
        x_logits, mu, logvar, logits = self.net(x_noisy, y if self.tcfg.sup_mode=="cvae" else None)
        recon = recon_loss(x_logits, clean, self.tcfg.recon)
        beta_t = self.tcfg.beta * min(1.0, max(0.0, epoch_idx / max(1, self.tcfg.kl_warmup_epochs)))
        kld = kl_loss(mu, logvar)
        sup = torch.tensor(0.0, device=x.device)
        acc = 0.0
        if self.net.aux_head_on and logits is not None:
            sup = F.cross_entropy(logits, y)
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                acc = float((pred == y).float().mean().item())
        loss = recon + beta_t * kld + self.tcfg.lambda_sup * sup
        return loss, acc

    @torch.no_grad()
    def _eval_epoch(self, loader, epoch_idx: int):
        self.net.eval(); tot, n = 0.0, 0; acc_tot = 0.0; n_acc = 0
        for x, y in loader:
            l, acc = self._sup_batch(x, y, epoch_idx, eval_only=True)
            bsz = x.size(0)
            tot += float(l.item()) * bsz; n += bsz
            if self.net.aux_head_on:
                acc_tot += acc * bsz; n_acc += bsz
        val = tot / max(n, 1)
        acc_val = (acc_tot / max(n_acc, 1)) if self.net.aux_head_on else 0.0
        return val, acc_val

    def fit(self, dl_train, dl_val, max_epochs=50):
        es = 0
        for e in range(max_epochs):
            self.net.train()
            for x, y in dl_train:
                self.optim.zero_grad(set_to_none=True)
                loss, _ = self._sup_batch(x, y, epoch_idx=e, eval_only=False)
                loss.backward()
                if self.tcfg.grad_clip is not None:
                    nn.utils.clip_grad_norm_(self.net.parameters(), self.tcfg.grad_clip)
                self.optim.step()
            val, acc_val = self._eval_epoch(dl_val, epoch_idx=e); self.epochs_done += 1
            improved = (val + 1e-12 < self.best - 0.0) and (acc_val + self.tcfg.min_acc_drop >= self.best_acc if self.net.aux_head_on else True)
            if improved:
                self.best = val; self.best_acc = acc_val
                self.best_state = {k: v.detach().cpu().clone() for k, v in self.net.state_dict().items()}
                es = 0
            else:
                es += 1
            if es >= self.tcfg.es_patience:
                break
        if self.best_state is not None:
            self.net.load_state_dict(self.best_state)
        return self.best, self.best_acc

# ==============================
# Snapshot / restore / guards
# ==============================

def snapshot(net: ConvVAE_Sup):
    return {"state": {k: v.detach().cpu().clone() for k, v in net.state_dict().items()},
            "channels": net.channels.copy(),
            "downs": net.downs.copy()}

def restore(net: ConvVAE_Sup, snap):
    if net.channels != snap["channels"] or net.downs != snap["downs"]:
        net.channels = snap["channels"].copy()
        net.downs = snap["downs"].copy()
        net._rebuild()
    net.load_state_dict(snap["state"])

def can_widen(net: ConvVAE_Sup, ex_k: int, scfg) -> bool:
    if ex_k <= 0: return False
    if any(c + ex_k > scfg.max_width for c in net.channels): return False
    projected = net.total_neurons() + ex_k * len(net.channels) * (net.enc_feat_h ** 2)
    return projected <= scfg.max_neurons

def can_deepen(net: ConvVAE_Sup, scfg) -> bool:
    if len(net.channels) + 1 > scfg.max_depth: return False
    projected = net.total_neurons() + net.channels[-1] * (net.enc_feat_h ** 2)
    return projected <= scfg.max_neurons

# ==============================
# Six ADP searchers
# ==============================

def _train_eval(net: ConvVAE_Sup, dl_train, dl_val, tcfg: TrainConfig, max_epochs: int) -> Tuple[float, float]:
    return InnerTrainer(net, tcfg, net.num_classes).fit(dl_train, dl_val, max_epochs=max_epochs)

def width_to_depth(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val, best_acc = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
    width_fails = 0
    while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v, a = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val, best_acc = v, a; best_snap = snapshot(net)
            depth_fails = 0
            while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
                pre2 = snapshot(net); net.append_depth()
                v2, a2 = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val, best_acc = v2, a2; best_snap = snapshot(net)
                else: depth_fails += 1; restore(net, pre2)
        else:
            width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def depth_to_width(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val, best_acc = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
    depth_fails = 0
    while depth_fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v, a = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta:
            best_val, best_acc = v, a; best_snap = snapshot(net)
            width_fails = 0
            while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
                pre2 = snapshot(net); net.widen_all(scfg.ex_k)
                v2, a2 = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
                if v2 < best_val - scfg.delta: best_val, best_acc = v2, a2; best_snap = snapshot(net)
                else: width_fails += 1; restore(net, pre2)
        else:
            depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def alt_depth_first(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val, best_acc = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v, a = _train_eval(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val, best_acc = v, a; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v, a = _train_eval(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val, best_acc = v, a; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def alt_width_first(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val, best_acc = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
    total = 0; ok = lambda e: scfg.max_total_epochs is None or e < scfg.max_total_epochs
    improved = True
    while improved and ok(total):
        improved = False
        width_fails = 0
        while width_fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg) and ok(total):
            pre = snapshot(net); net.widen_all(scfg.ex_k)
            v, a = _train_eval(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val, best_acc = v, a; best_snap = snapshot(net); improved = True
            else: width_fails += 1; restore(net, pre)
        depth_fails = 0
        while depth_fails < scfg.patience_depth and can_deepen(net, scfg) and ok(total):
            pre = snapshot(net); net.append_depth()
            v, a = _train_eval(net, dl_train, dl_val, tcfg, max_epochs); total += max_epochs
            if v < best_val - scfg.delta: best_val, best_acc = v, a; best_snap = snapshot(net); improved = True
            else: depth_fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def depth_only(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val, best_acc = _train_eval(net, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_depth and can_deepen(net, scfg):
        pre = snapshot(net); net.append_depth()
        v, a = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val, best_acc = v, a; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net

def width_only(net, dl_train, dl_val, tcfg, scfg, max_epochs=30):
    best_snap = snapshot(net); best_val, best_acc = _train_eval(net, dl_train, dl_val, tcfg, max_epochs); fails = 0
    while fails < scfg.patience_width and can_widen(net, scfg.ex_k, scfg):
        pre = snapshot(net); net.widen_all(scfg.ex_k)
        v, a = _train_eval(net, dl_train, dl_val, tcfg, max_epochs)
        if v < best_val - scfg.delta: best_val, best_acc = v, a; best_snap = snapshot(net)
        else: fails += 1; restore(net, pre)
    restore(net, best_snap); return net

# ==============================
# Builders
# ==============================

def build_vae(in_ch, img_size, init_width, init_depth, z_dim, num_classes, dropout,
              down_every=2, recon="bce", sup_mode="aux", aux_head=True, cvae_cond_ch=16):
    channels = [init_width] * init_depth
    downs = [(i % down_every == (down_every-1)) for i in range(init_depth)]
    net = ConvVAE_Sup(in_ch, img_size, channels, z_dim, num_classes, downs, dropout=dropout, recon=recon,
                      sup_mode=sup_mode, cvae_cond_ch=cvae_cond_ch, aux_head=aux_head)
    return net
