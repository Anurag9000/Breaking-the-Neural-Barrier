import argparse
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Make repo root importable
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from DAE.Supervised.Models.adp_dae_supervised import ADPDAE  # noqa: E402
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


def train_epoch(model: nn.Module, loader, optimizer, crit, device, corruption_std: float) -> float:
    model.train()
    tot, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        noisy = x + corruption_std * torch.randn_like(x) if corruption_std > 0 else x
        noisy = noisy.clamp(-1.0, 1.0)
        rec = model(noisy)
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
def val_epoch(model: nn.Module, loader, crit, device, corruption_std: float) -> float:
    model.eval()
    tot, n = 0.0, 0
    for x, _ in loader:
        x = x.to(device)
        noisy = x + corruption_std * torch.randn_like(x) if corruption_std > 0 else x
        noisy = noisy.clamp(-1.0, 1.0)
        rec = model(noisy)
        loss = crit(rec, x)
        b = x.size(0)
        tot += loss.item() * b
        n += b
    return tot / max(n, 1)


def main():
    p = argparse.ArgumentParser(description="ADP Supervised DAE (CIFAR10) runner")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--epochs-per-step", type=int, default=2, help="epochs per ADP evaluation step")
    p.add_argument("--corruption-std", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--policy", type=str, default="width_only",
                   choices=["width_only", "depth_only", "w2d", "d2w", "alt_w", "alt_d"])
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-depth", type=int, default=12)
    p.add_argument("--max-width", type=int, default=256)
    p.add_argument("--max-neurons", type=int, default=1_000_000)
    p.add_argument("--outdir", type=str, default="runs/supervised_dae")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dl_train, dl_val = make_loaders(args.data_root, args.batch_size, args.val_split)

    model = ADPDAE(in_channels=3, widths=[64, 64, 64, 64], pooling_indices=[])
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    crit = nn.MSELoss(reduction="mean")

    def train_fn():
        return train_epoch(model, dl_train, optimizer, crit, device, args.corruption_std)

    def val_fn():
        return val_epoch(model, dl_val, crit, device, args.corruption_std)

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

    # Save
    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    ckpt_path = os.path.join(args.outdir, f"adp_dae_supervised_{args.policy}.pt")
    torch.save({"model": model.state_dict(), "policy": args.policy}, ckpt_path)
    print(f"Saved: {ckpt_path}")


if __name__ == "__main__":
    main()
