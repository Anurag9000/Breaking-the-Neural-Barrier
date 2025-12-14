import os
import math
import time
import json
from dataclasses import dataclass

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn as nn
import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn.functional as F
from torch.utils.data import random_split, DataLoader
from torchvision import datasets, transforms, utils

from ddpm_classcond_unet import DDPMClassCondUNet

# -----------------------------
# Utilities
# -----------------------------

def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def beta_schedule_linear(n_steps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_steps)


def to_device(batch, device):
    x, y = batch
    x = x.to(device)
    y = y.to(device)
    return x, y


def make_cifar10_loaders(data_root: str, batch_size: int, vfrac: float = 0.1, seed: int = 42):
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform_train)
    n_val = int(len(full) * vfrac)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=g)
    # ensure val transform
    val_set.dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=transform_val)

    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


# -----------------------------
# Training
# -----------------------------
@dataclass
class TrainConfig:
    data_root: str = "./data"
    out_dir: str = "results_ddpm_classcond_unet"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    batch_size: int = 128
    n_steps: int = 1000  # diffusion steps
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_epochs: int = 200
    early_patience: int = 20
    grad_clip: float = 1.0
    cfg_dropout_prob: float = 0.1  # classifier-free guidance via label drop during training


def loss_eps_pred(model, x0, y, betas, device):
    """Standard DDPM epsilon prediction loss.
    Sample t ~ Uniform{0..T-1}; form noisy xt; predict eps_t; MSE.
    """
    T = betas.shape[0]
    b = x0.size(0)
    t_idx = torch.randint(0, T, (b,), device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)

    a_t = alphas_cumprod[t_idx]
    sqrt_a_t = torch.sqrt(a_t)
    sqrt_one_minus_a_t = torch.sqrt(1 - a_t)
    eps = torch.randn_like(x0)
    xt = sqrt_a_t[:, None, None, None] * x0 + sqrt_one_minus_a_t[:, None, None, None] * eps

    # Classifier-free guidance training: randomly drop labels to a null label
    drop_mask = (torch.rand_like(y.float()) < cfg.cfg_dropout_prob)
    y_train = y.clone()
    y_train[drop_mask] = 10  # null label index (num_classes == 10)

    t_float = (t_idx.float() + 0.5) / T
    eps_pred = model(xt, t_float, y_train)
    return F.mse_loss(eps_pred, eps)


def evaluate(model, loader, betas, device):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = to_device(batch, device)
            loss = loss_eps_pred(model, x, y, betas, device)
            bs = x.size(0)
            loss_sum += loss.item() * bs
            n += bs
    return loss_sum / max(1, n)


def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, val_loader, test_loader = make_cifar10_loaders(cfg.data_root, cfg.batch_size)

    model = DDPMClassCondUNet(num_classes=10)
    model.to(device)

    betas = beta_schedule_linear(cfg.n_steps).to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optim = torch.optim.AdamW(params, lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = {"val": float('inf'), "epoch": -1}
    patience = 0

    history = []
    global cfg  # for label drop in loss
    globals()["cfg"] = cfg


    # Init Logger


    logger = ContinuousLogger(Path('results_run_ddpm_classcond_unet'), 'run_ddpm_classcond_unet', 'train')


    for epoch in range(cfg.max_epochs):
        model.train()
        t0 = time.time()
        for batch in train_loader:
            x, y = to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            loss = loss_eps_pred(model, x, y, betas, device)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
        val_loss = evaluate(model, val_loader, betas, device)
        history.append({"epoch": epoch, "val_loss": val_loss})

        improved = val_loss < best["val"] - 1e-4
        if improved:
            best["val"], best["epoch"] = val_loss, epoch
            patience = 0
            torch.save({"model": model.state_dict(), "betas": betas.cpu()}, os.path.join(cfg.out_dir, "best.pth"))
        else:
            patience += 1
        # Log

        msg = f"Epoch {epoch} | val {val_loss:.4f} | best {best['val']:.4f} @ {best['epoch']} | {time.time(

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })-t0:.1f}s")
        if patience >= cfg.early_patience:
            break

    with open(os.path.join(cfg.out_dir, "history.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Sampling grid from the best model
    ckpt = torch.load(os.path.join(cfg.out_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt["model"]) 
    betas = ckpt["betas"].to(device)

    model.eval()
    classes = torch.arange(10, device=device)
    y = classes.repeat_interleave(8)  # 8 images per class
    with torch.no_grad():
        imgs = model.sample(
            shape=(y.numel(), 3, 32, 32),
            num_steps=cfg.n_steps,
            betas=betas,
            y=y,
            cfg_scale=2.0,  # single-model CFG
            device=device,
            eta=0.0,
        )
    grid = utils.make_grid((imgs + 1) * 0.5, nrow=8, padding=2)
    utils.save_image(grid, os.path.join(cfg.out_dir, "samples_grid.png"))


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)