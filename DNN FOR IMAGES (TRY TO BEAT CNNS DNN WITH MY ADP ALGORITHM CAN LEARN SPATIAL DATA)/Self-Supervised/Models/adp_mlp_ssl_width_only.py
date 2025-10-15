
import copy, math, os, time, random
from typing import List, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split

# ---------- Data utils ----------
class TwoCropTransform:
    def __init__(self, base_transform):
        self.base = base_transform
    def __call__(self, x):
        return self.base(x), self.base(x)

def build_ssl_transforms(img_size):
    blur_kernel = int(0.1*min(img_size))
    if blur_kernel % 2 == 0: blur_kernel += 1
    augmentation = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(0.4, 0.4, 0.2, 0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.GaussianBlur(kernel_size=blur_kernel, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
    ])
    return TwoCropTransform(augmentation)

def build_data(dataset, data_dir, img_size, val_split):
    tfm = build_ssl_transforms(img_size)
    dataset = dataset.lower()
    if dataset == "mnist":
        ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
        C=1
    elif dataset == "fashionmnist":
        ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
        C=1
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
        C=3
    elif dataset == "cifar100":
        ds = datasets.CIFAR100(data_dir, train=True, download=True, transform=tfm)
        C=3
    else:
        raise ValueError("Unsupported dataset")
    val_len = int(len(ds) * val_split)
    tr_len = len(ds) - val_len
    tr, va = random_split(ds, [tr_len, val_len])
    return tr, va, (C, img_size[0], img_size[1])

# ---------- Model ----------
class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bn: bool=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None: x = self.bn(x)
        return self.act(x)

def _resize_linear(old: nn.Linear, new_in: int, new_out: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out)
    with torch.no_grad():
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

class AdaptiveMLPSSL(nn.Module):
    def __init__(self, in_dim: int, hidden_widths: List[int], rep_dim: int, proj_dim: int, use_bn: bool=True):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_widths = list(hidden_widths)
        self.rep_dim = int(rep_dim)
        self.proj_dim = int(proj_dim)
        self.use_bn = use_bn
        self._build_modules()
        self.global_epoch = 0

    def _build_modules(self):
        # encoder
        enc_layers = []
        prev = self.in_dim
        for w in self.hidden_widths:
            enc_layers.append(MLPBlock(prev, w, self.use_bn))
            prev = w
        self.enc = nn.Sequential(*enc_layers)
        self.rep = nn.Linear(prev, self.rep_dim)

        # projector
        self.proj_fc1 = nn.Linear(self.rep_dim, self.rep_dim)
        self.proj_act = nn.ReLU(inplace=True)
        self.proj_fc2 = nn.Linear(self.rep_dim, self.proj_dim)

    def forward(self, img):
        x = img.view(img.size(0), -1)
        h = self.enc(x)
        z = self.rep(h)
        p = self.proj_fc2(self.proj_act(self.proj_fc1(z)))
        return z, p

    # metrics
    def total_neurons(self):
        return sum(self.hidden_widths) + self.rep_dim

    def depth(self):
        return len(self.hidden_widths) + 1  # hidden layers + rep layer

    # snapshot/restore
    def snapshot(self) -> Dict:
        return {
            "state": {k: v.detach().cpu() for k, v in self.state_dict().items()},
            "hidden": list(self.hidden_widths),
            "rep_dim": int(self.rep_dim),
            "proj_dim": int(self.proj_dim),
        }

    def restore(self, snap: Dict):
        self.hidden_widths = list(snap["hidden"])
        self.rep_dim = int(snap["rep_dim"])
        self.proj_dim = int(snap["proj_dim"])
        self._build_modules()
        self.load_state_dict(snap["state"], strict=True)

    # expansions
    def append_depth(self):
        new_w = self.hidden_widths[-1] if len(self.hidden_widths) > 0 else max(256, self.rep_dim)
        self.hidden_widths.append(int(new_w))

        old_state = copy.deepcopy(self.state_dict())
        old_modules = {"enc": self.enc, "rep": self.rep, "proj_fc1": self.proj_fc1, "proj_fc2": self.proj_fc2}
        self._build_modules()

        # transplant encoder blocks
        for i, (old_blk, new_blk) in enumerate(zip(old_modules["enc"], self.enc)):
            new_blk.linear = _resize_linear(old_blk.linear, new_blk.linear.in_features, new_blk.linear.out_features)
            if old_blk.bn is not None and new_blk.bn is not None:
                new_blk.bn = _resize_bn1d(old_blk.bn, new_blk.bn.num_features)
            self.enc[i] = new_blk
        # rep & projector
        self.rep = _resize_linear(old_modules["rep"], self.rep.in_features, self.rep.out_features)
        self.proj_fc1 = _resize_linear(old_modules["proj_fc1"], self.proj_fc1.in_features, self.proj_fc1.out_features)
        self.proj_fc2 = _resize_linear(old_modules["proj_fc2"], self.proj_fc2.in_features, self.proj_fc2.out_features)

    def widen_all(self, ex_k: int):
        self.hidden_widths = [int(w + ex_k) for w in self.hidden_widths]
        self.rep_dim = int(self.rep_dim + ex_k)

        old = {"enc": self.enc, "rep": self.rep, "proj_fc1": self.proj_fc1, "proj_fc2": self.proj_fc2}
        self._build_modules()

        for i, (old_blk, new_blk) in enumerate(zip(old["enc"], self.enc)):
            new_blk.linear = _resize_linear(old_blk.linear, new_blk.linear.in_features, new_blk.linear.out_features)
            if old_blk.bn is not None and new_blk.bn is not None:
                new_blk.bn = _resize_bn1d(old_blk.bn, new_blk.bn.num_features)
            self.enc[i] = new_blk

        self.rep = _resize_linear(old["rep"], self.rep.in_features, self.rep.out_features)
        self.proj_fc1 = _resize_linear(old["proj_fc1"], self.proj_fc1.in_features, self.proj_fc1.out_features)
        self.proj_fc2 = _resize_linear(old["proj_fc2"], self.proj_fc2.in_features, self.proj_fc2.out_features)

    # training utils
    @staticmethod
    def nt_xent_loss(p_i, p_j, temperature: float=0.2):
        z_i = F.normalize(p_i, dim=1)
        z_j = F.normalize(p_j, dim=1)
        N = z_i.size(0)
        z = torch.cat([z_i, z_j], dim=0)  # (2N,D)
        sim = torch.mm(z, z.t())  # (2N,2N)
        diag = torch.eye(2*N, device=z.device, dtype=torch.bool)
        sim.masked_fill_(diag, -9e15)
        pos = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
        logits = sim / temperature
        labels = pos
        loss = F.cross_entropy(logits, labels)
        return loss

    def train_inner(self, train_loader, val_loader, device, epochs: int, lr: float, patience: int, temperature: float=0.2):
        self.to(device)
        opt = optim.Adam(self.parameters(), lr=lr)

        best_val = float("inf")
        best_state = None
        bad = 0

        for _ in range(epochs):
            self.global_epoch += 1
            # train
            self.train()
            tr_sum, tr_n = 0.0, 0
            for (x1, x2), _ in train_loader:
                x1 = x1.to(device, non_blocking=True)
                x2 = x2.to(device, non_blocking=True)
                _, p1 = self(x1)
                _, p2 = self(x2)
                loss = self.nt_xent_loss(p1, p2, temperature=temperature)
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                opt.step()
                tr_sum += loss.item() * x1.size(0)
                tr_n += x1.size(0)

            # val
            self.eval()
            va_sum, va_n = 0.0, 0
            with torch.no_grad():
                for (x1, x2), _ in val_loader:
                    x1 = x1.to(device, non_blocking=True)
                    x2 = x2.to(device, non_blocking=True)
                    _, p1 = self(x1)
                    _, p2 = self(x2)
                    loss = self.nt_xent_loss(p1, p2, temperature=temperature)
                    va_sum += loss.item() * x1.size(0)
                    va_n += x1.size(0)

            tr = tr_sum / max(tr_n, 1)
            va = va_sum / max(va_n, 1)

            improved = va < best_val
            if improved:
                best_val = va
                best_state = {k: v.detach().cpu() for k, v in self.state_dict().items()}
                bad = 0
            else:
                bad += 1

            print(f"[{self.global_epoch:04d}] train_ntxent={tr:.6f} val_ntxent={va:.6f} | depth={self.depth()} total_neurons={self.total_neurons()} widths={self.hidden_widths}+[rep={self.rep_dim}]")

            if bad >= patience:
                break

        return best_val, best_state

def adp_search_width_only(model: AdaptiveMLPSSL, train_loader, val_loader, device,
                          trials_width: int, epochs: int, lr: float, patience: int,
                          delta: float, ex_k: int, max_neurons: int=None, max_depth: int=None, max_width: int=None,
                          temperature: float=0.2):
    best_val, best_state = model.train_inner(train_loader, val_loader, device, epochs, lr, patience, temperature)
    print(f"Initial val_ntxent={best_val:.6f}")
    w_trials = 0
    while w_trials < trials_width:
        if max_neurons is not None and model.total_neurons() >= max_neurons:
            print("Hit max_neurons; stopping.")
            break
        snap = model.snapshot()
        model.widen_all(ex_k=ex_k)
        if max_width is not None and max(model.hidden_widths+[model.rep_dim]) > max_width:
            print("Width step would exceed max_width; restoring.")
            model.restore(snap)
            break
        if max_depth is not None and model.depth() > max_depth:
            print("Width step invalidated depth guard; restoring.")
            model.restore(snap)
            break
        val, state = model.train_inner(train_loader, val_loader, device, epochs, lr, patience, temperature)
        if val + delta < best_val:
            print(f"ACCEPT width++ | {best_val:.6f} -> {val:.6f}")
            best_val, best_state = val, state
        else:
            print(f"REJECT width++ | {val:.6f} (>= {best_val:.6f} - delta)")
            model.restore(snap)
        w_trials += 1

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    return best_val
