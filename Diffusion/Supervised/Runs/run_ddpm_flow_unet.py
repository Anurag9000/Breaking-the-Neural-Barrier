from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.ddpm_flow_unet import FlowUNet
from runs._common_public_benchmarks import FlyingChairsTripletDataset


def main() -> None:
    root = Path(os.environ.get("BBNB_FLYINGCHAIRS_ROOT", "./data/FlyingChairs"))
    dataset = FlyingChairsTripletDataset(root=root, split="train", image_size=128)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = FlowUNet().to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        last_loss = None
        for pair, flow in loader:
            pair = pair.to(device)
            flow = flow.to(device)
            t = torch.randint(0, cfg.T, (pair.size(0),), device=device)
            loss = lossf(model, flow, pair, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
