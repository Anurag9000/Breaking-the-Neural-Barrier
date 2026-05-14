import argparse
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

from model_simclr_vit import SimCLRViT
from Transformer.Supervised.Runs._common_real_image import make_real_image_loaders
from utils.adp_logging import ContinuousLogger


@dataclass
class Config:
    data_root: str = "./data"
    img_size: int = 224
    patch_size: int = 16
    embed_dim: int = 384
    depth: int = 6
    heads: int = 6
    mlp_ratio: float = 4.0
    proj_hidden: int = 2048
    proj_out: int = 128
    temperature: float = 0.2
    batch_size: int = 256
    epochs: int = 400
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 30
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    out_dir: str = "./outs_simclr_vit"
    seed: int = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _make_tensor_aug(size):
    return transforms.Compose([
        transforms.RandomResizedCrop(size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.RandomGrayscale(p=0.2),
    ])


def _make_two_views(imgs, aug):
    x1 = imgs
    x2 = torch.stack([aug(img) for img in imgs])
    return x1, x2


def save_plot(curve, title, path, semilogy=False):
    import matplotlib.pyplot as plt

    plt.figure()
    plt.semilogy(curve) if semilogy else plt.plot(curve)
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, bbox_inches="tight")
    plt.close()


def train(cfg: Config):
    set_seed(cfg.seed)
    train_loader, val_loader, _ = make_real_image_loaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        image_size=cfg.img_size,
    )
    aug = _make_tensor_aug(cfg.img_size)
    model = SimCLRViT(
        cfg.img_size, cfg.patch_size, cfg.embed_dim, cfg.depth, cfg.heads, cfg.mlp_ratio, cfg.proj_hidden, cfg.proj_out
    ).to(cfg.device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    best_state = None
    patience = cfg.patience
    tr_curve = []
    va_curve = []
    logger = ContinuousLogger(Path("results_run_simclr_vit"), "run_simclr_vit", "train")

    for epoch in range(cfg.epochs):
        model.train()
        tr_loss = 0.0
        for imgs, _ in train_loader:
            x1, x2 = _make_two_views(imgs.to(cfg.device), aug)
            z1, z2 = model(x1, x2)
            loss = model.nt_xent(z1, z2, cfg.temperature)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss += loss.item() * imgs.size(0)
        tr_loss /= len(train_loader.dataset)
        tr_curve.append(tr_loss)

        model.eval()
        va_loss = 0.0
        with torch.no_grad():
            for imgs, _ in val_loader:
                x1, x2 = _make_two_views(imgs.to(cfg.device), aug)
                z1, z2 = model(x1, x2)
                loss = model.nt_xent(z1, z2, cfg.temperature)
                va_loss += loss.item() * imgs.size(0)
        va_loss /= len(val_loader.dataset)
        va_curve.append(va_loss)

        logger.log_console(f"Epoch {epoch+1}/{cfg.epochs} | train {tr_loss:.4f} | val {va_loss:.4f}")
        logger.log_epoch_stats({"epoch": epoch, "val_loss": va_loss, "train_loss": tr_loss})
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
    save_plot(tr_curve, "SimCLR Train Loss", os.path.join(cfg.out_dir, "train_loss.png"), semilogy=True)
    save_plot(va_curve, "SimCLR Val Loss", os.path.join(cfg.out_dir, "val_loss.png"), semilogy=True)

    if best_state is not None:
        torch.save(best_state, os.path.join(cfg.out_dir, "best_state.pt"))
        with open(os.path.join(cfg.out_dir, "summary.json"), "w") as f:
            json.dump({"best_val": float(best_val)}, f, indent=2)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=str, default="./data")
    p.add_argument("--img_size", type=int, default=224)
    p.add_argument("--patch_size", type=int, default=16)
    p.add_argument("--embed_dim", type=int, default=384)
    p.add_argument("--depth", type=int, default=6)
    p.add_argument("--heads", type=int, default=6)
    p.add_argument("--mlp_ratio", type=float, default=4.0)
    p.add_argument("--proj_hidden", type=int, default=2048)
    p.add_argument("--proj_out", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--patience", type=int, default=30)
    p.add_argument("--out_dir", type=str, default="./outs_simclr_vit")
    p.add_argument("--seed", type=int, default=42)
    cfg = Config(**vars(p.parse_args()))
    train(cfg)
