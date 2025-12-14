import os
import json
from dataclasses import dataclass

import torch
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_logging import ContinuousLogger.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils

from ddpm_segcond_unet import SegCondDDPMUNet


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def beta_schedule_linear(n_steps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_steps)


def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


def make_cifar10_loaders(data_root: str, batch_size: int, vfrac: float = 0.1, seed: int = 42):
    norm = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    tf_val = transforms.Compose([transforms.ToTensor(), norm])
    full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    n_val = int(len(full) * vfrac)
    n_train = len(full) - n_val
    g = torch.Generator().manual_seed(seed)
    train_set, val_set = random_split(full, [n_train, n_val], generator=g)
    val_set.dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tf_val)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_val)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=2, pin_memory=True)
    return train_loader, val_loader, test_loader


def seg_onehot_from_labels(y, H, W, K=10, device=None):
    # y: (B,) -> one-hot (B,K,H,W) broadcasting per pixel
    device = device or y.device
    b = y.size(0)
    oh = torch.zeros((b, K), device=device)
    oh.scatter_(1, y.view(-1,1), 1.0)
    oh = oh.view(b, K, 1, 1).expand(b, K, H, W)
    return oh


@dataclass
class TrainConfig:
    data_root: str = "./data"
    out_dir: str = "results_ddpm_segcond_unet"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    batch_size: int = 128
    n_steps: int = 1000
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_epochs: int = 200
    early_patience: int = 20
    grad_clip: float = 1.0


def eps_loss(model, x0, y, betas, device):
    T = betas.shape[0]
    b, _, H, W = x0.shape
    t_idx = torch.randint(0, T, (b,), device=device)
    alphas = 1.0 - betas
    ac = torch.cumprod(alphas, dim=0)
    a_t = ac[t_idx]
    sqrt_a = torch.sqrt(a_t)
    sqrtm = torch.sqrt(1 - a_t)
    eps = torch.randn_like(x0)
    xt = sqrt_a[:, None, None, None] * x0 + sqrtm[:, None, None, None] * eps

    t_float = (t_idx.float() + 0.5) / T
    seg = seg_onehot_from_labels(y, H, W, K=10, device=device)
    eps_pred = model(xt, t_float, seg)
    return F.mse_loss(eps_pred, eps)


def evaluate(model, loader, betas, device):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = to_device(batch, device)
            loss = eps_loss(model, x, y, betas, device)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
    return loss_sum / max(1, n)


def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, val_loader, test_loader = make_cifar10_loaders(cfg.data_root, cfg.batch_size)

    model = SegCondDDPMUNet(K=10)
    model.to(device)

    betas = beta_schedule_linear(cfg.n_steps).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = {"val": float('inf'), "epoch": -1}
    patience = 0
    hist = []


    # Init Logger


    logger = ContinuousLogger(Path('results_run_ddpm_segcond_unet'), 'run_ddpm_segcond_unet', 'train')


    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in train_loader:
            x, y = to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            loss = eps_loss(model, x, y, betas, device)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
        val = evaluate(model, val_loader, betas, device)
        hist.append({"epoch": epoch, "val_loss": val})
        if val < best['val'] - 1e-4:
            best['val'], best['epoch'] = val, epoch
            patience = 0
            torch.save({"model": model.state_dict(), "betas": betas.cpu()}, os.path.join(cfg.out_dir, "best.pth"))
        else:
            patience += 1
        # Log

        msg = f"Epoch {epoch} | val {val:.4f} | best {best['val']:.4f}@{best['epoch']}"

        logger.log_console(msg)

        logger.log_epoch_stats({

            "epoch": epoch,

            "val_loss": val_loss if 'val_loss' in locals() else (loss.item() if 'loss' in locals() else 0),

            "train_loss": loss.item() if 'loss' in locals() else 0

        })
        if patience >= cfg.early_patience:
            break

    with open(os.path.join(cfg.out_dir, "history.json"), 'w') as f:
        json.dump(hist, f, indent=2)

    # sample
    ckpt = torch.load(os.path.join(cfg.out_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt['model'])
    betas = ckpt['betas'].to(device)

    model.eval()
    batch = next(iter(test_loader))
    x0, y = to_device(batch, device)
    seg = seg_onehot_from_labels(y, x0.shape[-2], x0.shape[-1], K=10, device=device)
    with torch.no_grad():
        imgs = model.sample(seg, betas=betas, device=device, eta=0.0)
    grid = utils.make_grid((imgs + 1) * 0.5, nrow=8, padding=2)
    utils.save_image(grid, os.path.join(cfg.out_dir, "samples_grid.png"))


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)