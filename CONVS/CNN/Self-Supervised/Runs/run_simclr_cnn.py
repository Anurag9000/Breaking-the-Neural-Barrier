"""
Self-supervised training script for SimCLR + CNN encoder (CIFAR-10 by default)
----------------------------------------------------------------------------- 
Features
  • Strong SimCLR augmentations returning two views per sample
  • Mixed precision (AMP) training
  • AdamW optimizer by default (switchable to SGD)
  • Checkpointing: saves encoder+projector
  • Optional linear evaluation on CIFAR-10 with frozen encoder (--linear-eval)
  • Pool index convention: 0-based (e.g., --pool-after "0,2")

Usage (single-GPU):
  python run_simclr_cnn.py --data ./data --epochs 200 --batch-size 256 \
      --width 64 --depth 4 --pool-after "1,3" --temperature 0.5 \
      --proj-dim 128 --lr 3e-4 --wd 1e-4 --save-dir ./ckpt_simclr

Linear eval after SSL pretrain:
  python run_simclr_cnn.py --linear-eval --load ./ckpt_simclr/best.pth
"""
from __future__ import annotations
import argparse
import os
import time
from dataclasses import dataclass
from typing import Tuple, Optional, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

from pathlib import Path

# Local imports (when placed alongside this script)
from simclr_cnn_model import ConvEncoder, SimCLR, nt_xent_loss, total_neurons  # rename file to match import


# ----------------------------
# Data: SimCLR pair dataset
# ----------------------------
class GaussianBlur(object):
    """Implements ImageNet-style Gaussian blur for small images."""
    def __init__(self, p: float = 0.5, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return T.functional.gaussian_blur(x, kernel_size=[self.kernel_size, self.kernel_size], sigma=self.sigma)
        return x


def simclr_transform(image_size: int = 32) -> T.Compose:
    color_jitter = T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([color_jitter], p=0.8),
        T.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.5, kernel_size=3),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


class SimCLRPairDataset(Dataset):
    def __init__(self, base: Dataset, t1: T.Compose, t2: Optional[T.Compose] = None):
        self.base = base
        self.t1 = t1
        self.t2 = t2 or t1
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        x1 = self.t1(img)
        x2 = self.t2(img)
        return x1, x2


# ----------------------------
# Training & evaluation
# ----------------------------
@dataclass
class Args:
    data: str = "./data"
    save_dir: str = "./ckpt_simclr"
    load: Optional[str] = None
    epochs: int = 200
    batch_size: int = 256
    lr: float = 3e-4
    wd: float = 1e-4
    temperature: float = 0.5
    proj_dim: int = 128
    proj_hidden: Optional[int] = None
    width: int = 64
    depth: int = 4
    pool_after: str = "1,3"  # 0-based indices as string
    num_workers: int = 4
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    linear_eval: bool = False
    eval_epochs: int = 25
    eval_lr: float = 0.1
    eval_wd: float = 0.0


def set_seed(seed: int):
    import random, numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_pool(s: str) -> List[int]:
    s = s.strip()
    if not s:
        return []
    return [int(x) for x in s.split(',') if x.strip() != ""]


def build_dataloaders(args: Args):
    tf = simclr_transform(image_size=32)
    base_train = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=T.ToPILImage())
    ssl_train = SimCLRPairDataset(base_train, t1=tf)
    train_loader = DataLoader(ssl_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True, drop_last=True)

    # For linear eval: standard eval transform
    eval_tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    base_test = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=eval_tf)
    test_loader = DataLoader(base_test, batch_size=512, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    return train_loader, test_loader


def build_model(args: Args) -> SimCLR:
    pool_after = parse_pool(args.pool_after)
    encoder = ConvEncoder(in_ch=3, width=args.width, depth=args.depth, pool_after=pool_after, gap=True)
    model = SimCLR(encoder=encoder, proj_hidden=args.proj_hidden, proj_dim=args.proj_dim)
    return model


def save_checkpoint(model: SimCLR, optimizer: optim.Optimizer, scaler: GradScaler, epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
    }, path)


def train(args: Args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, test_loader = build_dataloaders(args)
    model = build_model(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_loss = float('inf')
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (x1, x2) in enumerate(train_loader, start=1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                z1, z2 = model(x1, x2)
                loss = nt_xent_loss(z1, z2, temperature=args.temperature)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if step % 50 == 0:
                avg = running / 50
                running = 0.0
                print(f"[Epoch {epoch:03d}] step {step:04d}/{len(train_loader)}  loss={avg:.4f}")

        # Epoch summary
        epoch_loss = evaluate_ssl_loss(model, train_loader, device, args.temperature)
        print(f"Epoch {epoch:03d} done | SSL loss: {epoch_loss:.4f} | neurons: {total_neurons(model):,}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, scaler, epoch, os.path.join(args.save_dir, 'best.pth'))

    print(f"Training finished in {time.time()-start:.1f}s. Best SSL loss: {best_loss:.4f}")

    if args.linear_eval:
        acc = linear_eval(args, model.encoder, test_loader, device)
        print(f"Linear eval (frozen encoder) accuracy: {acc*100:.2f}%")


def evaluate_ssl_loss(model: SimCLR, loader: DataLoader, device: torch.device, temperature: float) -> float:
    model.eval()
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for x1, x2 in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            z1, z2 = model(x1, x2)
            loss = nt_xent_loss(z1, z2, temperature=temperature)
            bsz = x1.size(0)
            total_loss += loss.item() * bsz
            n += bsz
    return total_loss / max(n, 1)


def linear_eval(args: Args, encoder: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    """Train a linear classifier on top of the frozen encoder using CIFAR-10 train set, evaluate on test set.
       (Simple, fast sanity check — not a full evaluation protocol.)
    """
    # Build train set with labels
    eval_tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_set = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=eval_tf)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=args.num_workers, pin_memory=True)

    # Freeze encoder
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    # Logistic regression head
    feat_dim = encoder.out_dim
    clf = nn.Linear(feat_dim, 10).to(device)
    optimizer = optim.SGD(clf.parameters(), lr=args.eval_lr, weight_decay=args.eval_wd, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    # Train a few epochs
    for epoch in range(1, args.eval_epochs + 1):
        clf.train()
        for x, y in train_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            with torch.no_grad():
                f = encoder(x)
                f = torch.nn.functional.normalize(f, dim=-1)
            logits = clf(f)
            loss = criterion(logits, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    # Evaluate
    clf.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            f = encoder(x)
            f = torch.nn.functional.normalize(f, dim=-1)
            logits = clf(f)
            pred = logits.argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.numel()
    return correct / max(1, total)


# ----------------------------
# CLI
# ----------------------------

def parse_args() -> Args:
    p = argparse.ArgumentParser()
    p.add_argument('--data', type=str, default='./data')
    p.add_argument('--save-dir', type=str, default='./ckpt_simclr')
    p.add_argument('--load', type=str, default=None)
    p.add_argument('--epochs', type=int, default=200)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=3e-4)
    p.add_argument('--wd', type=float, default=1e-4)
    p.add_argument('--temperature', type=float, default=0.5)
    p.add_argument('--proj-dim', type=int, default=128)
    p.add_argument('--proj-hidden', type=int, default=None)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--pool-after', type=str, default='1,3', help='0-based indices, e.g., "0,2"')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--linear-eval', action='store_true')
    p.add_argument('--eval-epochs', type=int, default=25)
    p.add_argument('--eval-lr', type=float, default=0.1)
    p.add_argument('--eval-wd', type=float, default=0.0)
    args = p.parse_args()
    return Args(**vars(args))


def main():
    args = parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    if args.load and os.path.isfile(args.load):
        ckpt = torch.load(args.load, map_location='cpu')
        model = build_model(args)
        model.load_state_dict(ckpt['model'])
        device = torch.device(args.device)
        model = model.to(device)
        print(f"Loaded checkpoint from {args.load}")
        if args.linear_eval:
            _, test_loader = build_dataloaders(args)
            acc = linear_eval(args, model.encoder, test_loader, device)
            print(f"Linear eval (frozen encoder) accuracy: {acc*100:.2f}%")
        else:
            print("--load was provided but --linear-eval not set; nothing to do.")
        return

    train(args)


if __name__ == '__main__':
    main()
