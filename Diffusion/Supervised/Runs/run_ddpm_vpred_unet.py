import os
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torchvision import utils

from ddpm_vpred_unet import DDPMVPredUNet
from runs._common_real_image import infer_num_classes, make_real_image_loaders


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def beta_schedule_cosine(n_steps: int = 1000):
    s = 0.008
    steps = torch.arange(n_steps + 1)
    f = torch.cos(((steps / n_steps) + s) / (1 + s) * 0.5 * torch.pi) ** 2
    alphas_cumprod = f / f[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return betas.clamp(1e-5, 0.999)


def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


@dataclass
class TrainConfig:
    data_root: str = "./data"
    out_dir: str = "results_ddpm_vpred_unet"
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
    num_classes: int = 10


def v_loss(model, x0, y, alphas_cumprod, device, cfg):
    T = alphas_cumprod.shape[0]
    t_idx = torch.randint(0, T, (x0.size(0),), device=device)
    a_t = alphas_cumprod[t_idx]
    sqrt_a = torch.sqrt(a_t)
    sqrtm = torch.sqrt(1 - a_t)
    eps = torch.randn_like(x0)
    xt = sqrt_a[:, None, None, None] * x0 + sqrtm[:, None, None, None] * eps

    drop = torch.rand_like(y.float()) < cfg.cfg_dropout_prob
    y_train = y.clone()
    y_train[drop] = cfg.num_classes
    t_float = (t_idx.float() + 0.5) / T
    v_pred = model(xt, t_float, y_train)
    v_target = sqrt_a[:, None, None, None] * eps - sqrtm[:, None, None, None] * x0
    return F.mse_loss(v_pred, v_target)


def evaluate(model, loader, alphas_cumprod, device, cfg):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = to_device(batch, device)
            loss = v_loss(model, x, y, alphas_cumprod, device, cfg)
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
    cfg.num_classes = infer_num_classes(train_loader)

    model = DDPMVPredUNet(num_classes=cfg.num_classes).to(device)
    betas = beta_schedule_cosine(cfg.n_steps)
    alphas_cumprod = torch.cumprod(1.0 - betas, dim=0).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    patience = 0
    best_path = os.path.join(cfg.out_dir, "best.pth")

    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in train_loader:
            x, y = to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            loss = v_loss(model, x, y, alphas_cumprod, device, cfg)
            loss.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
        val = evaluate(model, val_loader, alphas_cumprod, device, cfg)
        print(f"epoch {epoch}: val {val:.4f}")
        if val < best_val - 1e-4:
            best_val = val
            patience = 0
            torch.save({"model": model.state_dict(), "alphas_cumprod": alphas_cumprod.cpu()}, best_path)
        else:
            patience += 1
        if patience >= cfg.early_patience:
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    alphas_cumprod = ckpt["alphas_cumprod"].to(device)

    model.eval()
    classes = torch.arange(cfg.num_classes, device=device)
    y = classes.repeat_interleave(8)
    with torch.no_grad():
        imgs = model.sample((y.numel(), 3, 32, 32), alphas_cumprod=alphas_cumprod, y=y, cfg_scale=2.0, device=device)
    grid = torch.clamp((imgs + 1) * 0.5, 0, 1)
    utils.save_image(grid, os.path.join(cfg.out_dir, "samples_grid.png"), nrow=8, padding=2)


if __name__ == "__main__":
    train(TrainConfig())
