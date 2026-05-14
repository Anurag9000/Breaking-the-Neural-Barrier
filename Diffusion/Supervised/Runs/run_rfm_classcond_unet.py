import os
from dataclasses import dataclass

import torch
from torchvision import utils

from rfm_classcond_unet import RectifiedFlowUNet
from runs._common_real_image import infer_num_classes, make_real_image_loaders


def set_seed(seed: int = 42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_device(batch, device):
    x, y = batch
    return x.to(device), y.to(device)


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


def rfm_loss(model, data_x0, y, device, cfg, num_classes):
    x1 = torch.randn_like(data_x0)
    t = torch.rand(data_x0.size(0), device=device)
    x_t = (1 - t)[:, None, None, None] * data_x0 + t[:, None, None, None] * x1
    drop = torch.rand_like(y.float()) < cfg.cfg_dropout_prob
    y_train = y.clone()
    y_train[drop] = num_classes
    v_pred = model(x_t, t, y_train)
    v_target = x1 - data_x0
    return torch.nn.functional.mse_loss(v_pred, v_target)


def evaluate(model, loader, device, cfg, num_classes):
    model.eval()
    loss_sum, n = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            x, y = to_device(batch, device)
            l = rfm_loss(model, x, y, device, cfg, num_classes)
            loss_sum += l.item() * x.size(0)
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

    model = RectifiedFlowUNet(num_classes=num_classes).to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    best_val = float("inf")
    patience = 0
    best_path = os.path.join(cfg.out_dir, "best.pth")

    for epoch in range(cfg.max_epochs):
        model.train()
        for batch in train_loader:
            x, y = to_device(batch, device)
            optim.zero_grad(set_to_none=True)
            l = rfm_loss(model, x, y, device, cfg, num_classes)
            l.backward()
            if cfg.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            optim.step()
        val = evaluate(model, val_loader, device, cfg, num_classes)
        print(f"epoch {epoch}: val {val:.4f}")
        if val < best_val - 1e-4:
            best_val = val
            patience = 0
            torch.save({"model": model.state_dict(), "num_classes": num_classes}, best_path)
        else:
            patience += 1
        if patience >= cfg.early_patience:
            break

    ckpt = torch.load(best_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    num_classes = int(ckpt["num_classes"])

    model.eval()
    classes = torch.arange(num_classes, device=device)
    y = classes.repeat_interleave(8)
    x1 = torch.randn((y.numel(), 3, 32, 32), device=device)
    with torch.no_grad():
        imgs = model.sample(x1, steps=1000, y=y, device=device)
    grid = torch.clamp((imgs + 1) * 0.5, 0, 1)
    utils.save_image(grid, os.path.join(cfg.out_dir, "samples_grid.png"), nrow=8, padding=2)


if __name__ == "__main__":
    train(TrainConfig())

