import argparse
import os
import random
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Make repo root importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from DAE.Unsupervised.Models.adp_dae_unsupervised import ADPDenoisingConvAE  # noqa: E402
from DAE.adp_search import (  # noqa: E402
    SearchConfig,
    search_width_only,
    search_depth_only,
    search_width_to_depth,
    search_depth_to_width,
    search_alt_width_first,
    search_alt_depth_first,
)


def make_loaders(data_root: str, batch_size: int, val_split: float = 0.1, num_workers: int = 4, download: bool = True):
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    ds_full = datasets.CIFAR10(root=data_root, train=True, transform=tf, download=download)
    n_val = int(len(ds_full) * val_split)
    n_train = len(ds_full) - n_val
    g = torch.Generator().manual_seed(0)
    ds_train, ds_val = random_split(ds_full, [n_train, n_val], generator=g)
    dl_train = DataLoader(ds_train, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    dl_val = DataLoader(ds_val, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return dl_train, dl_val


def _corrupt_gaussian(x: torch.Tensor, std: float) -> torch.Tensor:
    if std <= 0:
        return x
    noise = torch.randn_like(x) * std
    return (x + noise).clamp(-1.0, 1.0)


def _pixel_mask(x: torch.Tensor, p: float) -> torch.Tensor:
    if p <= 0:
        return torch.zeros_like(x)
    B, _, H, W = x.shape
    return (torch.rand(B, 1, H, W, device=x.device) < p).float()


def _patch_mask(x: torch.Tensor, ratio: float, patch: int) -> torch.Tensor:
    if ratio <= 0:
        return torch.zeros_like(x)
    B, _, H, W = x.shape
    gh, gw = H // patch, W // patch
    pm = (torch.rand(B, 1, gh, gw, device=x.device) < ratio).float()
    return F.interpolate(pm, size=(H, W), mode="nearest")


def _hole_mask(x: torch.Tensor, holes: int, min_frac: float, max_frac: float, rng: random.Random) -> torch.Tensor:
    B, _, H, W = x.shape
    m = torch.zeros(B, 1, H, W, device=x.device)
    area = H * W
    for b in range(B):
        for _ in range(max(1, holes)):
            target = rng.uniform(min_frac, max_frac) * area
            aspect = rng.uniform(0.5, 2.0)
            h = int(round((target * aspect) ** 0.5))
            w = int(round((target / aspect) ** 0.5))
            h = max(1, min(H, h))
            w = max(1, min(W, w))
            y0 = rng.randint(0, max(0, H - h)) if H - h > 0 else 0
            x0 = rng.randint(0, max(0, W - w)) if W - w > 0 else 0
            m[b, 0, y0 : y0 + h, x0 : x0 + w] = 1.0
    return m


def corrupt(x: torch.Tensor, args) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns corrupted input and an optional mask indicating supervised pixels.
    For blindspot/inpaint we return a mask to compute masked loss.
    """
    mode = args.mode
    if mode == "gaussian":
        return _corrupt_gaussian(x, args.std), None
    elif mode == "pixel_mask":
        mask = _pixel_mask(x, args.mask_prob)
        return x * (1.0 - mask), None
    elif mode == "patch_mask":
        mask = _patch_mask(x, args.patch_ratio, args.patch_size)
        return x * (1.0 - mask), None
    elif mode == "blindspot":
        mask = _pixel_mask(x, args.mask_prob)
        return x * (1.0 - mask), mask
    elif mode == "inpaint":
        mask = _hole_mask(x, args.holes_per_image, args.min_hole_frac, args.max_hole_frac, args._rng)
        return x * (1.0 - mask), mask
    elif mode == "energy":
        # energy mode uses gaussian noise as corruption by default
        return _corrupt_gaussian(x, args.std), None
    else:
        return x, None


def train_epoch(model: nn.Module, loader, optimizer, crit, device, cargs) -> float:
    model.train()
    tot, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        noisy, mask = corrupt(x, cargs)
        rec = model(noisy)

        if cargs.mode == "blindspot":
            diff = (rec - x) ** 2
            loss = (diff * mask).sum() / (mask.sum() + 1e-8)
        elif cargs.mode == "inpaint":
            diff = (rec - x) ** 2
            loss = (diff * mask).sum() / (mask.sum() + 1e-8)
        elif cargs.mode == "energy":
            # positive energy
            E_pos = crit(rec, x)
            # negative sample
            if cargs.energy_neg_mode == "roll":
                x_neg = x.roll(shifts=1, dims=0)
            elif cargs.energy_neg_mode == "cutout":
                m = _hole_mask(x, cargs.holes_per_image, cargs.min_hole_frac, cargs.max_hole_frac, cargs._rng)
                x_neg = x * (1.0 - m)
            else:
                x_neg = _corrupt_gaussian(x, cargs.energy_std)
            rec_neg = model(x_neg)
            E_neg = crit(rec_neg, x_neg)
            loss = E_pos + 0.25 * torch.relu(cargs.energy_margin - (E_neg - E_pos))
        else:
            loss = crit(rec, x)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        b = x.size(0)
        tot += loss.item() * b
        n += b
    return tot / max(n, 1)


@torch.no_grad()
def val_epoch(model: nn.Module, loader, crit, device, cargs) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        noisy, mask = corrupt(x, cargs)
        rec = model(noisy)
        if cargs.mode in ("blindspot", "inpaint"):
            diff = (rec - x) ** 2
            loss = (diff * mask).sum() / (mask.sum() + 1e-8)
        elif cargs.mode == "energy":
            E_pos = crit(rec, x)
            if cargs.energy_neg_mode == "roll":
                x_neg = x.roll(shifts=1, dims=0)
            elif cargs.energy_neg_mode == "cutout":
                m = _hole_mask(x, cargs.holes_per_image, cargs.min_hole_frac, cargs.max_hole_frac, cargs._rng)
                x_neg = x * (1.0 - m)
            else:
                x_neg = _corrupt_gaussian(x, cargs.energy_std)
            rec_neg = model(x_neg)
            E_neg = crit(rec_neg, x_neg)
            loss = E_pos + 0.25 * torch.relu(cargs.energy_margin - (E_neg - E_pos))
        else:
            loss = crit(rec, x)

        b = x.size(0)
        tot += loss.item() * b
        n += b
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="ADP Self-Supervised DAE (CIFAR10) runner")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--epochs-per-step", type=int, default=2, help="epochs per ADP evaluation step")
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--policy", type=str, default="width_only",
                   choices=["width_only", "depth_only", "w2d", "d2w", "alt_w", "alt_d"])
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-width", type=int, default=256)
    p.add_argument("--max-neurons", type=int, default=1_000_000)
    # corruption flags
    p.add_argument("--mode", type=str, default="gaussian",
                   choices=["gaussian", "pixel_mask", "patch_mask", "blindspot", "inpaint", "energy", "none"])
    p.add_argument("--std", type=float, default=0.1)
    p.add_argument("--mask-prob", type=float, default=0.05)
    p.add_argument("--patch-ratio", type=float, default=0.6)
    p.add_argument("--patch-size", type=int, default=4)
    p.add_argument("--holes-per-image", type=int, default=1)
    p.add_argument("--min-hole-frac", type=float, default=0.15)
    p.add_argument("--max-hole-frac", type=float, default=0.35)
    p.add_argument("--energy-margin", type=float, default=0.05)
    p.add_argument("--energy-neg-mode", type=str, default="gaussian", choices=["gaussian", "roll", "cutout"])
    p.add_argument("--energy-std", type=float, default=0.15)
    p.add_argument("--outdir", type=str, default="runs/unsupervised_dae")
    args = p.parse_args()

    # rng for hole masks
    args._rng = random.Random(1234)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_train, dl_val = make_loaders(args.data_root, args.batch_size, args.val_split)

    model = ADPDenoisingConvAE(in_ch=3, widths=[16, 32, 64], pooling_indices=[0, 2])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.MSELoss(reduction="mean")

    def train_fn():
        return train_epoch(model, dl_train, optimizer, crit, device, args)

    def val_fn():
        return val_epoch(model, dl_val, crit, device, args)

    scfg = SearchConfig(ex_k=args.ex_k, max_depth=args.max_depth, max_width=args.max_width, max_neurons=args.max_neurons)

    if args.policy == "width_only":
        model = search_width_only(model, scfg, args.epochs_per_step, train_fn, val_fn)
    elif args.policy == "depth_only":
        model = search_depth_only(model, scfg, args.epochs_per_step, train_fn, val_fn)
    elif args.policy == "w2d":
        model = search_width_to_depth(model, scfg, args.epochs_per_step, train_fn, val_fn)
    elif args.policy == "d2w":
        model = search_depth_to_width(model, scfg, args.epochs_per_step, train_fn, val_fn)
    elif args.policy == "alt_w":
        model = search_alt_width_first(model, scfg, args.epochs_per_step, train_fn, val_fn)
    elif args.policy == "alt_d":
        model = search_alt_depth_first(model, scfg, args.epochs_per_step, train_fn, val_fn)

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, f"adp_dae_unsupervised_{args.policy}.pt")
    torch.save({"model": model.state_dict(), "policy": args.policy}, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
