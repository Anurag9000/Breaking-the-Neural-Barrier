import os
import math
import json
import random
import argparse
import numpy as np
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from dataclasses import dataclass
from model_mae_vit import MAEViT

# ===============
# Config
# ===============
@dataclass
class Config:
    data_root: str = './data'
    dataset: str = 'CIFAR10'  # CIFAR10 or CIFAR100
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    dec_embed_dim: int = 192
    dec_depth: int = 4
    dec_heads: int = 6
    mask_ratio: float = 0.75
    batch_size: int = 128
    epochs: int = 200
    lr: float = 1e-3
    weight_decay: float = 0.05
    patience: int = 20
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    out_dir: str = './outs_mae_vit'
    seed: int = 42

# ===============
# Utils
# ===============

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def make_loaders(cfg: Config):
    size = cfg.img_size
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(size),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
    ])
    if cfg.dataset == 'CIFAR10':
        full = datasets.CIFAR10(cfg.data_root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR10(cfg.data_root, train=False, download=True, transform=eval_tf)
    else:
        full = datasets.CIFAR100(cfg.data_root, train=True, download=True, transform=train_tf)
        test = datasets.CIFAR100(cfg.data_root, train=False, download=True, transform=eval_tf)

    n_val = int(0.1 * len(full))
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(cfg.seed)
    train, val = random_split(full, [n_train, n_val], generator=g)
    val.dataset.transform = eval_tf

    train_loader = DataLoader(train, batch_size=cfg.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test, batch_size=cfg.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return train_loader, val_loader, test_loader


def save_plot(curve, title, path, semilogy=False):
    plt.figure()
    if semilogy:
        plt.semilogy(curve)
    else:
        plt.plot(curve)
    plt.title(title)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.grid(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches='tight')
    plt.close()


# ===============
# Train MAE
# ===============

def train_mae(cfg: Config):
    set_seed(cfg.seed)
    train_loader, val_loader, test_loader = make_loaders(cfg)

    model = MAEViT(img_size=cfg.img_size,
                   patch_size=cfg.patch_size,
                   embed_dim=cfg.embed_dim,
                   depth=cfg.depth,
                   heads=cfg.heads,
                   mlp_ratio=cfg.mlp_ratio,
                   dec_embed_dim=cfg.dec_embed_dim,
                   dec_depth=cfg.dec_depth,
                   dec_heads=cfg.dec_heads,
                   mask_ratio=cfg.mask_ratio).to(cfg.device)

    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float('inf')
    best_state = None
    patience = cfg.patience

    tr_curve, va_curve = [], []


    # Init Logger


    logger = ContinuousLogger(Path('results_run_mae_vit'), 'run_mae_vit', 'train')


    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(cfg.device)
            pred, mask = model(imgs)
            loss = model.loss((pred, mask), imgs)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
        tr_loss /= len(train_loader.dataset)
        tr_curve.append(tr_loss)

        # val
        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                imgs = imgs.to(cfg.device)
                pred, mask = model(imgs)
                loss = model.loss((pred, mask), imgs)
                va_loss += loss.item() * imgs.size(0)
        va_loss /= len(val_loader.dataset)
        va_curve.append(va_loss)

        # Log


        msg = f"Epoch {epoch+1}/{cfg.epochs} | train {tr_loss:.4f} | val {va_loss:.4f}"


        logger.log_console(msg)


        logger.log_epoch_stats({


            "epoch": epoch,


            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),


            "train_loss": loss.item() if 'loss' in locals() else 0


        })

        if va_loss < best_val - 1e-4:
            best_val = va_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience = cfg.patience
        else:
            patience -= 1
            if patience == 0:
                print("Early stopping.")
                break

    os.makedirs(cfg.out_dir, exist_ok=True)
    save_plot(tr_curve, 'MAE Train Loss', os.path.join(cfg.out_dir, 'train_loss.png'), semilogy=True)
    save_plot(va_curve, 'MAE Val Loss', os.path.join(cfg.out_dir, 'val_loss.png'), semilogy=True)

    # Final test recon loss with best state
    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    te_loss = 0.0
    with torch.no_grad():
        for imgs, _ in test_loader:
            imgs = imgs.to(cfg.device)
            pred, mask = model(imgs)
            loss = model.loss((pred, mask), imgs)
            te_loss += loss.item() * imgs.size(0)
    te_loss /= len(test_loader.dataset)

    with open(os.path.join(cfg.out_dir, 'summary.json'), 'w') as f:
        json.dump({
            'best_val_loss': float(best_val),
            'test_loss': float(te_loss)
        }, f, indent=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--patch_size', type=int, default=16)
    parser.add_argument('--embed_dim', type=int, default=384)
    parser.add_argument('--depth', type=int, default=6)
    parser.add_argument('--heads', type=int, default=6)
    parser.add_argument('--mlp_ratio', type=float, default=4.0)
    parser.add_argument('--dec_embed_dim', type=int, default=192)
    parser.add_argument('--dec_depth', type=int, default=4)
    parser.add_argument('--dec_heads', type=int, default=6)
    parser.add_argument('--mask_ratio', type=float, default=0.75)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=20)
    parser.add_argument('--out_dir', type=str, default='./outs_mae_vit')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    cfg = Config(**vars(args))
    train_mae(cfg)