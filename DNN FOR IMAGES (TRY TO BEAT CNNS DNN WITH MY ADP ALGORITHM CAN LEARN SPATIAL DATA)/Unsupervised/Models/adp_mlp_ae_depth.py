
import copy, math, os, time, random
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# -------------------- Blocks & Model --------------------

class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bn: bool=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)

def _resize_linear(old: nn.Linear, new_in: int, new_out: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out)
    with torch.no_grad():
        # Copy overlap
        in_overlap = min(old.in_features, new_in)
        out_overlap = min(old.out_features, new_out)
        if in_overlap > 0 and out_overlap > 0:
            new.weight[:out_overlap, :in_overlap].copy_(old.weight[:out_overlap, :in_overlap])
            if old.bias is not None and new.bias is not None:
                new.bias[:out_overlap].copy_(old.bias[:out_overlap])
    return new

def _resize_bn1d(old: nn.BatchNorm1d, new_features: int) -> nn.BatchNorm1d:
    new = nn.BatchNorm1d(new_features)
    with torch.no_grad():
        overlap = min(old.num_features, new_features)
        if overlap > 0:
            new.weight[:overlap].copy_(old.weight[:overlap])
            new.bias[:overlap].copy_(old.bias[:overlap])
            new.running_mean[:overlap].copy_(old.running_mean[:overlap])
            new.running_var[:overlap].copy_(old.running_var[:overlap])
    return new

class AdaptiveMLPAE(nn.Module):
    """
    Single-model adaptive MLP Autoencoder for images.
    Architecture:
      input -> [hidden_widths...] -> bottleneck -> [reversed(hidden_widths)...] -> output
    Expansion ops:
      - append_depth(): adds one hidden layer (encoder+decoder) with the last hidden width
      - widen_all(k): increases every hidden width and bottleneck by k
    """
    def __init__(self, in_dim: int, hidden_widths: List[int], bottleneck: int, use_bn: bool=True, output_activation: str="sigmoid"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_widths = list(hidden_widths)
        self.bottleneck = int(bottleneck)
        self.use_bn = use_bn
        self.output_activation = output_activation

        self._build_modules()

        # global-epoch counter for logging parity with CNN versions
        self.global_epoch = 0

    # ---------- builder ----------
    def _build_modules(self):
        # Encoder
        enc_layers = []
        prev = self.in_dim
        for w in self.hidden_widths:
            blk = MLPBlock(prev, w, self.use_bn)
            enc_layers.append(blk)
            prev = w
        self.enc = nn.Sequential(*enc_layers)
        self.fc_mu = nn.Linear(prev, self.bottleneck)

        # Decoder (mirror)
        dec_layers = []
        prev = self.bottleneck
        for w in reversed(self.hidden_widths):
            blk = MLPBlock(prev, w, self.use_bn)
            dec_layers.append(blk)
            prev = w
        self.dec = nn.Sequential(*dec_layers)
        self.out = nn.Linear(prev, self.in_dim)

    def encode(self, x):
        return self.fc_mu(self.enc(x))

    def decode(self, z):
        x = self.dec(z)
        x = self.out(x)
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = torch.tanh(x)
        return x

    def forward(self, img):
        x = img.view(img.size(0), -1)
        z = self.encode(x)
        xr = self.decode(z)
        return xr

    # ---------- capacity metrics ----------
    def total_neurons(self) -> int:
        return sum(self.hidden_widths) + self.bottleneck

    def depth(self) -> int:
        # number of encoder hidden layers + decoder hidden layers + bottleneck counted as 1
        return len(self.hidden_widths)*2 + 1

    # ---------- snapshot / restore ----------
    def snapshot(self) -> Dict:
        return {
            "state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "hidden": list(self.hidden_widths),
            "bottleneck": int(self.bottleneck),
        }

    def restore(self, snap: Dict):
        self.hidden_widths = list(snap["hidden"])
        self.bottleneck = int(snap["bottleneck"])
        self._build_modules()
        self.load_state_dict(snap["state"], strict=True)

    # ---------- expansion operators ----------
    def append_depth(self):
        if len(self.hidden_widths) == 0:
            new_w = max(32, self.bottleneck)  # safe default
        else:
            new_w = self.hidden_widths[-1]
        self.hidden_widths.append(int(new_w))

        # Rebuild while transplanting weights
        old = copy.deepcopy(self.state_dict())
        old_hidden = self.hidden_widths[:-1]  # before append
        old_bottleneck = self.bottleneck

        self._build_modules()  # rebuild with new layer

        # transplant overlap
        sd = self.state_dict()
        for k in sd.keys():
            if k in old:
                try:
                    sd[k][:old[k].shape[0]] = old[k]
                except Exception:
                    sd[k] = old[k]
        self.load_state_dict(sd, strict=False)

    def widen_all(self, ex_k: int):
        self.hidden_widths = [int(w + ex_k) for w in self.hidden_widths]
        self.bottleneck = int(self.bottleneck + ex_k)

        # Transplant weights layer-by-layer
        old_modules = {
            "enc": self.enc,
            "fc_mu": self.fc_mu,
            "dec": self.dec,
            "out": self.out,
        }
        old_state = copy.deepcopy(self.state_dict())

        # rebuild with new widths
        self._build_modules()

        # copy overlap for encoder blocks
        for i, (old_blk, new_blk) in enumerate(zip(old_modules["enc"], self.enc)):
            new_blk.linear = _resize_linear(old_blk.linear, new_blk.linear.in_features, new_blk.linear.out_features)
            if old_blk.bn is not None and new_blk.bn is not None:
                new_blk.bn = _resize_bn1d(old_blk.bn, new_blk.bn.num_features)
            # reassign to module
            self.enc[i] = new_blk

        # fc_mu
        self.fc_mu = _resize_linear(old_modules["fc_mu"], self.fc_mu.in_features, self.fc_mu.out_features)

        # copy overlap for decoder blocks
        for i, (old_blk, new_blk) in enumerate(zip(old_modules["dec"], self.dec)):
            new_blk.linear = _resize_linear(old_blk.linear, new_blk.linear.in_features, new_blk.linear.out_features)
            if old_blk.bn is not None and new_blk.bn is not None:
                new_blk.bn = _resize_bn1d(old_blk.bn, new_blk.bn.num_features)
            self.dec[i] = new_blk

        # out
        self.out = _resize_linear(old_modules["out"], self.out.in_features, self.out.out_features)

    # ---------- training utilities ----------
    def _step(self, imgs, optimizer, loss_fn, denoise_std: float=0.0):
        self.train()
        imgs = imgs
        clean = imgs
        if denoise_std > 0:
            noisy = torch.clamp(imgs + denoise_std * torch.randn_like(imgs), 0.0, 1.0)
        else:
            noisy = imgs
        optimizer.zero_grad()
        recon = self(noisy)
        loss = loss_fn(recon, clean.view(clean.size(0), -1))
        loss.backward()
        nn.utils.clip_grad_norm_(self.parameters(), 1.0)
        optimizer.step()
        return loss.detach()

    @torch.no_grad()
    def _eval_step(self, imgs, loss_fn, denoise_std: float=0.0):
        self.eval()
        clean = imgs
        if denoise_std > 0:
            noisy = torch.clamp(imgs + denoise_std * torch.randn_like(imgs), 0.0, 1.0)
        else:
            noisy = imgs
        recon = self(noisy)
        loss = loss_fn(recon, clean.view(clean.size(0), -1))
        return loss.detach()

    def train_inner(self, train_loader, val_loader, device, epochs: int, lr: float, patience: int, denoise_std: float=0.0):
        self.to(device)
        opt = optim.Adam(self.parameters(), lr=lr)
        loss_fn = nn.MSELoss()

        best_val = float("inf")
        best_state = None
        bad = 0

        for ep in range(1, epochs+1):
            self.global_epoch += 1
            # train
            self.train()
            tr_sum, tr_n = 0.0, 0
            for imgs, _ in train_loader:
                imgs = imgs.to(device, non_blocking=True)
                loss = self._step(imgs, opt, loss_fn, denoise_std=denoise_std)
                tr_sum += loss.item() * imgs.size(0)
                tr_n += imgs.size(0)

            # eval
            self.eval()
            va_sum, va_n = 0.0, 0
            for imgs, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                loss = self._eval_step(imgs, loss_fn, denoise_std=denoise_std)
                va_sum += loss.item() * imgs.size(0)
                va_n += imgs.size(0)

            tr = tr_sum / max(tr_n, 1)
            va = va_sum / max(va_n, 1)

            improved = va < best_val
            if improved:
                best_val = va
                best_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                bad = 0
            else:
                bad += 1

            print(f"[{self.global_epoch:04d}] train_mse={tr:.6f} val_mse={va:.6f} | depth={self.depth()} total_neurons={self.total_neurons()} widths={self.hidden_widths}+[{self.bottleneck}]")

            if bad >= patience:
                break

        return best_val, best_state

def adp_search_depth_then_width(model: AdaptiveMLPAE, train_loader, val_loader, device,
                                trials_depth: int, trials_width: int, epochs: int, lr: float, patience: int,
                                delta: float, ex_k: int, max_neurons: int=None, max_depth: int=None, max_width: int=None,
                                denoise_std: float=0.0):
    """
    Greedy search: try depth expansions first, then width expansions.
    Accept an expansion if best val MSE improves by at least `delta`.
    """
    # initial train
    best_val, best_state = model.train_inner(train_loader, val_loader, device, epochs, lr, patience, denoise_std)
    print(f"Initial val_mse={best_val:.6f}")

    # ----- Depth phase -----
    d_trials = 0
    while d_trials < trials_depth:
        if max_depth is not None and model.depth() >= max_depth:
            print("Hit max_depth; stopping depth phase.")
            break
        if max_neurons is not None and model.total_neurons() >= max_neurons:
            print("Hit max_neurons; stopping depth phase.")
            break

        snap = model.snapshot()
        model.append_depth()
        if max_width is not None and max(model.hidden_widths+[model.bottleneck]) > max_width:
            print("Depth step would exceed max_width; restoring.")
            model.restore(snap)
            break

        val, state = model.train_inner(train_loader, val_loader, device, epochs, lr, patience, denoise_std)
        if val + delta < best_val:
            print(f"ACCEPT depth++ | {best_val:.6f} -> {val:.6f}")
            best_val, best_state = val, state
        else:
            print(f"REJECT depth++ | {val:.6f} (>= {best_val:.6f} - delta)")
            model.restore(snap)
        d_trials += 1

    # ensure model is at best state so far
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    # ----- Width phase -----
    w_trials = 0
    while w_trials < trials_width:
        if max_neurons is not None and model.total_neurons() >= max_neurons:
            print("Hit max_neurons; stopping width phase.")
            break

        snap = model.snapshot()
        model.widen_all(ex_k=ex_k)
        if max_width is not None and max(model.hidden_widths+[model.bottleneck]) > max_width:
            print("Width step would exceed max_width; restoring.")
            model.restore(snap)
            break
        if max_depth is not None and model.depth() > max_depth:
            print("Width step invalidated depth guard; restoring.")
            model.restore(snap)
            break

        val, state = model.train_inner(train_loader, val_loader, device, epochs, lr, patience, denoise_std)
        if val + delta < best_val:
            print(f"ACCEPT width++ | {best_val:.6f} -> {val:.6f}")
            best_val, best_state = val, state
        else:
            print(f"REJECT width++ | {val:.6f} (>= {best_val:.6f} - delta)")
            model.restore(snap)
        w_trials += 1

    # load best at end
    if best_state is not None:
        model.load_state_dict(best_state, strict=True)

    return best_val
