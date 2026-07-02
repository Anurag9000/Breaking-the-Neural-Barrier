"""
Self-supervised training script for Barlow Twins + CNN encoder (CIFAR-10 by default)
-----------------------------------------------------------------------------------
Features
  • Strong SimCLR-style augmentations producing two views
  • 3-layer projector with BN (last BN without affine) as per Barlow Twins
  • AMP training, AdamW optimizer, optional cosine LR
  • Checkpointing of encoder+projector
  • Optional linear evaluation with frozen encoder (--linear-eval)

Usage:
  python run_barlowtwins_cnn.py --data ./data --epochs 300 --batch-size 256 \
      --width 64 --depth 4 --pool-after "1,3" --proj-dim 2048 --lambda 5e-3 \
      --lr 1e-3 --wd 1e-6 --save-dir ./ckpt_barlow

Linear eval after SSL pretrain:
  python run_barlowtwins_cnn.py --linear-eval --load ./ckpt_barlow/best.pth
"""
from __future__ import annotations
import argparse
import math
import os
import time
from dataclasses import dataclass
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Dataset
import torchvision
import torchvision.transforms as T

from pathlib import Path

# Local import (place barlowtwins_cnn_model.py next to this script)
from barlowtwins_cnn_model import BarlowTwins, ConvEncoder, barlow_twins_loss, total_neurons


# ----------------------------
# Data: augmentations & pair dataset
# ----------------------------
class GaussianBlur(object):
    def __init__(self, p: float = 0.5, kernel_size: int = 3, sigma: Tuple[float, float] = (0.1, 2.0)):
        self.p = p
        self.kernel_size = kernel_size
        self.sigma = sigma
    def __call__(self, x):
        if torch.rand(1).item() < self.p:
            return T.functional.gaussian_blur(x, kernel_size=[self.kernel_size, self.kernel_size], sigma=self.sigma)
        return x


def ssl_transform(image_size: int = 32) -> T.Compose:
    cj = T.ColorJitter(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2)
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([cj], p=0.8),
        T.RandomGrayscale(p=0.2),
        GaussianBlur(p=0.5, kernel_size=3),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


class PairDataset(Dataset):
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
    save_dir: str = "./ckpt_barlow"
    load: Optional[str] = None
    epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    wd: float = 1e-6
    cosine: bool = True
    lambd: float = 5e-3
    proj_dim: int = 2048
    proj_hidden: Optional[int] = None
    width: int = 64
    depth: int = 4
    pool_after: str = "1,3"      # 0-based indices
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    linear_eval: bool = False
    eval_epochs: int = 25
    eval_lr: float = 0.2
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
    tf = ssl_transform(image_size=32)
    base_train = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=T.ToPILImage())
    ssl_train = PairDataset(base_train, t1=tf)
    train_loader = DataLoader(ssl_train, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False, drop_last=True)

    eval_tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    base_test = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=eval_tf)
    test_loader = DataLoader(base_test, batch_size=512, shuffle=False, num_workers=args.num_workers, pin_memory=False)
    return train_loader, test_loader


def build_model(args: Args) -> BarlowTwins:
    pool_after = parse_pool(args.pool_after)
    encoder = ConvEncoder(in_ch=3, width=args.width, depth=args.depth, pool_after=pool_after, gap=True)
    model = BarlowTwins(encoder=encoder, proj_hidden=args.proj_hidden, proj_dim=args.proj_dim)
    return model


def save_checkpoint(model: BarlowTwins, optimizer: optim.Optimizer, scaler: GradScaler, epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
    }, path)


def cosine_scheduler(base_lr: float, step: int, total_steps: int) -> float:
    t = min(step / max(1, total_steps), 1.0)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))


def train(args: Args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, test_loader = build_dataloaders(args)
    model = build_model(args).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=(device.type == 'cuda'))

    best_loss = float('inf')
    global_step = 0
    total_steps = args.epochs * len(train_loader)
    start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for step, (x1, x2) in enumerate(train_loader, start=1):
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)

            if args.cosine:
                lr = cosine_scheduler(args.lr, global_step, total_steps)
                for pg in optimizer.param_groups:
                    pg['lr'] = lr

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                z1, z2 = model(x1, x2)
                loss = barlow_twins_loss(z1, z2, lambd=args.lambd)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            global_step += 1
            running += loss.item()
            if step % 50 == 0:
                avg = running / 50
                running = 0.0
                print(f"[Epoch {epoch:03d}] step {step:04d}/{len(train_loader)}  loss={avg:.4f}")

        # Epoch summary
        epoch_loss = evaluate_ssl_loss(model, train_loader, device, lambd=args.lambd, max_batches=5)
        print(f"Epoch {epoch:03d} | SSL loss: {epoch_loss:.4f} | neurons: {total_neurons(model):,}")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_checkpoint(model, optimizer, scaler, epoch, os.path.join(args.save_dir, 'best.pth'))

    print(f"Training finished in {time.time()-start:.1f}s. Best SSL loss: {best_loss:.4f}")

    if args.linear_eval:
        acc = linear_eval(args, model.encoder, test_loader, device)
        print(f"Linear eval (frozen encoder) accuracy: {acc*100:.2f}%")


def evaluate_ssl_loss(model: BarlowTwins, loader: DataLoader, device: torch.device, lambd: float, max_batches: int = 5) -> float:
    model.eval()
    total_loss, n_seen, batches = 0.0, 0, 0
    with torch.no_grad():
        for x1, x2 in loader:
            x1 = x1.to(device, non_blocking=True)
            x2 = x2.to(device, non_blocking=True)
            z1, z2 = model(x1, x2)
            loss = barlow_twins_loss(z1, z2, lambd=lambd)
            bsz = x1.size(0)
            total_loss += loss.item() * bsz
            n_seen += bsz
            batches += 1
            if batches >= max_batches:
                break
    return total_loss / max(1, n_seen)


def linear_eval(args: Args, encoder: nn.Module, test_loader: DataLoader, device: torch.device) -> float:
    # Build labeled CIFAR-10 train loader
    eval_tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    train_set = torchvision.datasets.CIFAR10(args.data, train=True, download=True, transform=eval_tf)
    train_loader = DataLoader(train_set, batch_size=512, shuffle=True, num_workers=args.num_workers, pin_memory=False)

    # Freeze encoder
    encoder = encoder.to(device).eval()
    for p in encoder.parameters():
        p.requires_grad = False

    feat_dim = encoder.out_dim
    clf = nn.Linear(feat_dim, 10).to(device)
    optimizer = optim.SGD(clf.parameters(), lr=args.eval_lr, weight_decay=args.eval_wd, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

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
    p.add_argument('--save-dir', type=str, default='./ckpt_barlow')
    p.add_argument('--load', type=str, default=None)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-6)
    p.add_argument('--cosine', action='store_true', default=True)
    p.add_argument('--no-cosine', dest='cosine', action='store_false')
    p.add_argument('--lambda', dest='lambd', type=float, default=5e-3)
    p.add_argument('--proj-dim', type=int, default=2048)
    p.add_argument('--proj-hidden', type=int, default=None)
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--pool-after', type=str, default='1,3', help='0-based indices, e.g., "0,2"')
    p.add_argument('--num-workers', type=int, default=4)
    p.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--linear-eval', action='store_true')
    p.add_argument('--eval-epochs', type=int, default=25)
    p.add_argument('--eval-lr', type=float, default=0.2)
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
            print("--load provided but --linear-eval not set; nothing else to do.")
        return

    train(args)


if __name__ == '__main__':
    main()
