from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.ddpm_regress_unet import RegressUNet
from runs._common_public_benchmarks import VocSegmentationDataset


def main() -> None:
    root = Path(os.environ.get("BBNB_VOC_ROOT", "./data/VOCdevkit"))
    dataset = VocSegmentationDataset(root=root, year=os.environ.get("BBNB_VOC_YEAR", "2012"), image_set="train", image_size=256, target_mode="mask")
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = RegressUNet(target_ch=1).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        last_loss = None
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            t = torch.randint(0, cfg.T, (x.size(0),), device=device)
            loss = lossf(model, y, x, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
