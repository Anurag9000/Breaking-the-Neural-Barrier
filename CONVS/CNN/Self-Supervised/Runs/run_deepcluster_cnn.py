"""
Self-supervised training script for DeepCluster + CNN encoder (CIFAR-10 by default)
----------------------------------------------------------------------------------
Procedure (classic loop)
  1) Extract features for all training images with the current encoder.
  2) Run spherical k-means to obtain K cluster assignments.
  3) Train a classifier head on top of frozen (or lightly updated) encoder to predict cluster IDs.
  4) Unfreeze encoder and jointly update a few epochs (optional), then repeat from 1).

This compact script alternates every --recluster-epochs epochs.

Usage:
  python run_deepcluster_cnn.py --data ./data --epochs 300 --batch-size 256 \
      --width 64 --depth 4 --pool-after "1,3" --K 100 --recluster-epochs 10 \
      --lr 1e-3 --wd 1e-6 --save-dir ./ckpt_deepcluster

After SSL pretrain, you can run linear eval by:
  python run_deepcluster_cnn.py --linear-eval --load ./ckpt_deepcluster/best.pth
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

# Local import (place deepcluster_cnn_model.py next to this script)
from deepcluster_cnn_model import DeepCluster, ConvEncoder, kmeans_cosine, total_neurons


# ----------------------------
# Data
# ----------------------------

def ssl_transform(image_size: int = 32) -> T.Compose:
    cj = T.ColorJitter(brightness=0.6, contrast=0.6, saturation=0.6, hue=0.2)
    return T.Compose([
        T.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomApply([cj], p=0.8),
        T.RandomGrayscale(p=0.2),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])


# Simple dataset that returns (img, index) to align cluster labels
class IndexDataset(Dataset):
    def __init__(self, base: Dataset, tf: T.Compose):
        self.base = base
        self.tf = tf
    def __len__(self):
        return len(self.base)
    def __getitem__(self, idx):
        img, _ = self.base[idx]
        return self.tf(img), idx


# ----------------------------
# Training & evaluation
# ----------------------------
@dataclass
class Args:
    data: str = "./data"
    save_dir: str = "./ckpt_deepcluster"
    load: Optional[str] = None
    epochs: int = 300
    batch_size: int = 256
    lr: float = 1e-3
    wd: float = 1e-6
    cosine: bool = True
    width: int = 64
    depth: int = 4
    pool_after: str = "1,3"      # 0-based indices
    K: int = 100
    recluster_epochs: int = 10
    freeze_encoder_during_head: bool = True
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
    train_ds = IndexDataset(base_train, tf=tf)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=False, drop_last=True)

    # Eval
    eval_tf = T.Compose([T.ToTensor(), T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])
    base_test = torchvision.datasets.CIFAR10(args.data, train=False, download=True, transform=eval_tf)
    test_loader = DataLoader(base_test, batch_size=512, shuffle=False, num_workers=args.num_workers, pin_memory=False)

    return train_loader, test_loader, len(base_train)


def build_model(args: Args) -> DeepCluster:
    pool_after = parse_pool(args.pool_after)
    encoder = ConvEncoder(in_ch=3, width=args.width, depth=args.depth, pool_after=pool_after, gap=True)
    model = DeepCluster(encoder=encoder, K=args.K)
    return model


def save_checkpoint(model: DeepCluster, optimizer: optim.Optimizer, scaler: GradScaler, epoch: int, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'epoch': epoch,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scaler': scaler.state_dict(),
    }, path)


def cosine_scheduler(base_lr: float, step: int, total_steps: int) -> float:
    import math
    t = min(step / max(1, total_steps), 1.0)
    return 0.5 * base_lr * (1 + math.cos(math.pi * t))


@torch.no_grad()
def extract_features(model: DeepCluster, loader: DataLoader, N: int, device: torch.device) -> torch.Tensor:
    model.eval()
    feats = torch.zeros((N, model.encoder.out_dim), dtype=torch.float32, device=device)
    for x, idx in loader:
        x = x.to(device, non_blocking=True)
        f = model.encode(x)  # normalized
        feats[idx] = f
    return feats


def train(args: Args):
    set_seed(args.seed)
    device = torch.device(args.device)

    train_loader, test_loader, N = build_dataloaders(args)
    model = build_model(args).to(device)

    # two optimizers: head-only (optionally freeze encoder), and joint
    opt_all = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    opt_head = optim.AdamW(model.head.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = GradScaler(enabled=(device.type == 'cuda'))
    criterion = nn.CrossEntropyLoss()

    # initialize random assignments
    assignments = torch.randint(low=0, high=args.K, size=(N,), device=device)

    best_loss = float('inf')
    global_step = 0

    for epoch in range(1, args.epochs + 1):
        # Reclustering step
        if (epoch - 1) % args.recluster_epochs == 0 or epoch == 1:
            feats = extract_features(model, train_loader, N, device)
            assignments, _ = kmeans_cosine(feats, K=args.K, iters=30, seed=args.seed)
            print(f"[DeepCluster] Reclustered at epoch {epoch}: K={args.K}")

        # Train for one epoch
        model.train()
        if args.freeze_encoder_during_head:
            for p in model.encoder.parameters():
                p.requires_grad = False
            optimizer = opt_head
        else:
            for p in model.encoder.parameters():
                p.requires_grad = True
            optimizer = opt_all

        running = 0.0
        for step, (x, idx) in enumerate(train_loader, start=1):
            x = x.to(device, non_blocking=True)
            y = assignments[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=(device.type == 'cuda')):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running += loss.item()
            if step % 50 == 0:
                avg = running / 50
                running = 0.0
                print(f"[Epoch {epoch:03d}] step {step:04d}/{len(train_loader)}  loss={avg:.4f}")

        # Optionally unfreeze and do a few joint epochs (simple variant: one extra epoch right after head)
        if args.freeze_encoder_during_head:
            for p in model.encoder.parameters():
                p.requires_grad = True
            # single joint pass
            for step, (x, idx) in enumerate(train_loader, start=1):
                x = x.to(device, non_blocking=True)
                y = assignments[idx].to(device, non_blocking=True)
                opt_all.zero_grad(set_to_none=True)
                with autocast(enabled=(device.type == 'cuda')):
                    logits = model(x)
                    loss = criterion(logits, y)
                scaler.scale(loss).backward()
                scaler.step(opt_all)
                scaler.update()
                if step % 200 == 0:
                    print(f"  (joint) step {step:04d}/{len(train_loader)}  loss={loss.item():.4f}")

        # quick loss over a small eval subset for model selection
        val = quick_eval_loss(model, train_loader, device, assignments, max_batches=5)
        print(f"Epoch {epoch:03d} | proxy loss: {val:.4f} | neurons: {total_neurons(model):,}")
        if val < best_loss:
            best_loss = val
            save_checkpoint(model, opt_all, scaler, epoch, os.path.join(args.save_dir, 'best.pth'))

    print(f"Training finished. Best proxy loss: {best_loss:.4f}")

    if args.linear_eval:
        acc = linear_eval(args, model.encoder, test_loader, device)
        print(f"Linear eval (frozen encoder) accuracy: {acc*100:.2f}%")


@torch.no_grad()
def quick_eval_loss(model: DeepCluster, loader: DataLoader, device: torch.device, assignments: torch.Tensor, max_batches: int = 5) -> float:
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction='sum')
    total, count = 0.0, 0
    for i, (x, idx) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        y = assignments[idx].to(device, non_blocking=True)
        logits = model(x)
        total += criterion(logits, y).item()
        count += x.size(0)
        if i + 1 >= max_batches:
            break
    return total / max(1, count)


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
    p.add_argument('--save-dir', type=str, default='./ckpt_deepcluster')
    p.add_argument('--load', type=str, default=None)
    p.add_argument('--epochs', type=int, default=300)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--wd', type=float, default=1e-6)
    p.add_argument('--cosine', action='store_true', default=True)
    p.add_argument('--no-cosine', dest='cosine', action='store_false')
    p.add_argument('--width', type=int, default=64)
    p.add_argument('--depth', type=int, default=4)
    p.add_argument('--pool-after', type=str, default='1,3', help='0-based indices, e.g., "0,2"')
    p.add_argument('--K', type=int, default=100)
    p.add_argument('--recluster-epochs', type=int, default=10)
    p.add_argument('--freeze-encoder-during-head', action='store_true', default=True)
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
            _, test_loader, _ = build_dataloaders(args)
            acc = linear_eval(args, model.encoder, test_loader, device)
            print(f"Linear eval (frozen encoder) accuracy: {acc*100:.2f}%")
        else:
            print("--load provided but --linear-eval not set; nothing else to do.")
        return

    train(args)


if __name__ == '__main__':
    main()
