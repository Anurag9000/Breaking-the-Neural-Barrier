import os
import time
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ddpm_classcond_unet import DDPMClassCondUNet
from runs._common_real_image import infer_num_classes, make_real_image_loaders


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def beta_schedule_linear(n_steps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_steps)


def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


@dataclass
class TrainConfig:
    data_root: str = "./data"
    out_dir: str = "results_ddpm_classcond_unet"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    batch_size: int = 128
    n_steps: int = 1000
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_epochs: int = 200
    early_patience: int = 20
    grad_clip: float = 1.0
    cfg_dropout_prob: float = 0.1


def loss_eps_pred(model, x0, y, betas, device, num_classes, cfg_dropout_prob):
    t_idx = torch.randint(0, betas.shape[0], (x0.size(0),), device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    a_t = alphas_cumprod[t_idx]
    sqrt_a_t = torch.sqrt(a_t)
    sqrt_one_minus_a_t = torch.sqrt(1 - a_t)
    eps = torch.randn_like(x0)
    xt = sqrt_a_t[:, None, None, None] * x0 + sqrt_one_minus_a_t[:, None, None, None] * eps

    drop_mask = torch.rand_like(y.float()) < cfg_dropout_prob
    y_train = y.clone()
    y_train[drop_mask] = num_classes

    t_float = (t_idx.float() + 0.5) / betas.shape[0]
    eps_pred = model(xt, t_float, y_train)
    return F.mse_loss(eps_pred, eps)


def evaluate(model, loader, betas, device, num_classes, cfg_dropout_prob):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = to_device(batch, device)
            loss = loss_eps_pred(model, x, y, betas, device, num_classes, cfg_dropout_prob)
            loss_sum += loss.item() * x.size(0)
            n += x.size(0)
    return loss_sum / max(1, n)


def train(cfg: TrainConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    set_seed(cfg.seed)
    device = torch.device(cfg.device)

    train_loader, val_loader, test_loader = make_real_image_loaders(
        data_root=cfg.data_root,
        batch_size=cfg.batch_size,
        image_size=32,
    )
    num_classes = infer_num_classes(train_loader)

    model = DDPMClassCondUNet(num_classes=num_classes).to(device)
    betas = beta_schedule_linear(cfg.n_steps).to(device)
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    best_val = float("inf")
    patience = 0
    best_path = os.path.join(cfg.out_dir, "best.pth")

    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in train_loader:
            x, y = to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            loss = loss_eps_pred(model, x, y, betas, device, num_classes, cfg.cfg_dropout_prob)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()

        val_loss = evaluate(model, val_loader, betas, device, num_classes, cfg.cfg_dropout_prob)
        print(f"epoch {epoch}: val {val_loss:.4f}")
        if val_loss < best_val - 1e-4:
            best_val = val_loss
            patience = 0
            torch.save({"model": model.state_dict(), "betas": betas.cpu(), "num_classes": num_classes}, best_path)
        else:
            patience += 1
        if patience >= cfg.early_patience:
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    betas = ckpt["betas"].to(device)
    num_classes = int(ckpt["num_classes"])

    model.eval()
    classes = torch.arange(num_classes, device=device)
    y = classes.repeat_interleave(8)
    with torch.no_grad():
        imgs = model.sample(
            shape=(y.numel(), 3, 32, 32),
            num_steps=cfg.n_steps,
            betas=betas,
            y=y,
            cfg_scale=2.0,
            device=device,
            eta=0.0,
        )
    grid = torch.clamp((imgs + 1) * 0.5, 0, 1)
    os.makedirs(cfg.out_dir, exist_ok=True)
    from torchvision import utils

    utils.save_image(grid, os.path.join(cfg.out_dir, "samples_grid.png"), nrow=8, padding=2)


if __name__ == "__main__":
    train(TrainConfig())

