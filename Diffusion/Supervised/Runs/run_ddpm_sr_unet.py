import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import utils

from ddpm_sr_unet import SRDDPMUNet
from runs._common_real_image import make_real_image_loaders


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def beta_schedule_linear(n_steps: int, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, n_steps)


def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


def make_lr_condition(x_hr):
    x = (x_hr + 1) * 0.5
    x_lr = F.interpolate(x, scale_factor=0.5, mode="bicubic", align_corners=False)
    x_up = F.interpolate(x_lr, size=x_hr.shape[-2:], mode="bicubic", align_corners=False)
    return x_up * 2 - 1


@dataclass
class TrainConfig:
    data_root: str = "./data"
    out_dir: str = "results_ddpm_sr_unet"
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    batch_size: int = 128
    n_steps: int = 1000
    lr: float = 2e-4
    weight_decay: float = 0.0
    max_epochs: int = 200
    early_patience: int = 20
    grad_clip: float = 1.0


def eps_loss(model, x0, betas, device):
    t_idx = torch.randint(0, betas.shape[0], (x0.size(0),), device=device)
    alphas = 1.0 - betas
    ac = torch.cumprod(alphas, dim=0)
    a_t = ac[t_idx]
    sqrt_a = torch.sqrt(a_t)
    sqrtm = torch.sqrt(1 - a_t)
    eps = torch.randn_like(x0)
    xt = sqrt_a[:, None, None, None] * x0 + sqrtm[:, None, None, None] * eps
    t_float = (t_idx.float() + 0.5) / betas.shape[0]
    cond = make_lr_condition(x0)
    eps_pred = model(xt, t_float, cond)
    return F.mse_loss(eps_pred, eps)


def evaluate(model, loader, betas, device):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, _ = to_device(batch, device)
            loss = eps_loss(model, x, betas, device)
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

    model = SRDDPMUNet().to(device)
    betas = beta_schedule_linear(cfg.n_steps).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    patience = 0
    best_path = os.path.join(cfg.out_dir, "best.pth")

    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in train_loader:
            x, _ = to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            loss = eps_loss(model, x, betas, device)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
        val = evaluate(model, val_loader, betas, device)
        print(f"epoch {epoch}: val {val:.4f}")
        if val < best_val - 1e-4:
            best_val = val
            patience = 0
            torch.save({"model": model.state_dict(), "betas": betas.cpu()}, best_path)
        else:
            patience += 1
        if patience >= cfg.early_patience:
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    betas = ckpt["betas"].to(device)

    model.eval()
    batch = next(iter(test_loader))
    x_hr, _ = to_device(batch, device)
    cond = make_lr_condition(x_hr)
    with torch.no_grad():
        imgs = model.sample(cond, betas=betas, device=device, eta=0.0)
    grid = torch.clamp((imgs + 1) * 0.5, 0, 1)
    utils.save_image(grid, os.path.join(cfg.out_dir, "samples_grid.png"), nrow=8, padding=2)


if __name__ == "__main__":
    train(TrainConfig())

