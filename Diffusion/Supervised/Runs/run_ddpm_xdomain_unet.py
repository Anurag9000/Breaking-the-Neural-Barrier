from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.ddpm_xdomain_unet import XDomainUNet
from runs._common_public_benchmarks import PairedImageFolderDataset


def main() -> None:
    source_root = Path(os.environ.get("BBNB_XDOMAIN_SOURCE_ROOT", "./data/domain_a"))
    target_root = Path(os.environ.get("BBNB_XDOMAIN_TARGET_ROOT", "./data/domain_b"))
    dataset = PairedImageFolderDataset(source_root=source_root, target_root=target_root, image_size=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = XDomainUNet(y_ch=3, x_ch=3).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        last_loss = None
        for x_src, y_tgt in loader:
            x_src = x_src.to(device)
            y_tgt = y_tgt.to(device)
            t = torch.randint(0, cfg.T, (x_src.size(0),), device=device)
            loss = lossf(model, y_tgt, x_src, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
