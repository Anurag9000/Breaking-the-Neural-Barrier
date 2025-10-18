import os
import json
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils

from rfm_classcond_unet import RectifiedFlowUNet


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


def make_cifar10_loaders(data_root: str, batch_size: int, vfrac: float = 0.1, seed: int = 42):
    tf_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
    tf_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])
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


@dataclass
class TrainConfig:
    data_root: str = "./data"
    out_dir: str = "results_rfm_classcond_unet"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_epochs: int = 200
    early_patience: int = 20
    grad_clip: float = 1.0
    cfg_dropout_prob: float = 0.1


def rfm_loss(model, data_x0, y, device, cfg):
    # Sample x1 ~ N(0, I), t ~ U(0,1); x_t = (1-t) x0 + t x1; target v = x1 - x0
    b = data_x0.size(0)
    x1 = torch.randn_like(data_x0)
    t = torch.rand(b, device=device)
    x_t = (1 - t)[:, None, None, None] * data_x0 + t[:, None, None, None] * x1

    drop = (torch.rand_like(y.float()) < cfg.cfg_dropout_prob)
    y_train = y.clone()
    y_train[drop] = 10

    v_pred = model(x_t, t, y_train)
    v_target = x1 - data_x0
    return F.mse_loss(v_pred, v_target)


def evaluate(model, loader, device, cfg):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = to_device(batch, device)
            l = rfm_loss(model, x, y, device, cfg)
            loss_sum += l.item() * x.size(0)
            n += x.size(0)
    return loss_sum / max(1, n)


def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, val_loader, test_loader = make_cifar10_loaders(cfg.data_root, cfg.batch_size)

    model = RectifiedFlowUNet(num_classes=10)
    model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best = {"val": float('inf'), "epoch": -1}
    patience = 0
    hist = []

    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in train_loader:
            x, y = to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            l = rfm_loss(model, x, y, device, cfg)
            l.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
        val = evaluate(model, val_loader, device, cfg)
        hist.append({"epoch": epoch, "val_loss": val})
        if val < best['val'] - 1e-4:
            best['val'], best['epoch'] = val, epoch
            patience = 0
            torch.save({"model": model.state_dict()}, os.path.join(cfg.out_dir, "best.pth"))
        else:
            patience += 1
        print(f"Epoch {epoch} | val {val:.4f} | best {best['val']:.4f}@{best['epoch']}")
        if patience >= cfg.early_patience:
            break

    with open(os.path.join(cfg.out_dir, "history.json"), 'w') as f:
        json.dump(hist, f, indent=2)

    # sample
    ckpt = torch.load(os.path.join(cfg.out_dir, "best.pth"), map_location=device)
    model.load_state_dict(ckpt['model'])

    model.eval()
    classes = torch.arange(10, device=device)
    y = classes.repeat_interleave(8)
    x1 = torch.randn((y.numel(), 3, 32, 32), device=device)
    with torch.no_grad():
        imgs = model.sample(x1, steps=1000, y=y, device=device)
    grid = utils.make_grid((imgs + 1) * 0.5, nrow=8, padding=2)
    utils.save_image(grid, os.path.join(cfg.out_dir, "samples_grid.png"))


if __name__ == "__main__":
    cfg = TrainConfig()
    train(cfg)
