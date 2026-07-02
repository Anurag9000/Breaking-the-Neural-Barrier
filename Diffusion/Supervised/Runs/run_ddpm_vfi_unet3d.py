from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.ddpm_vfi_unet3d import VFIUNet
from runs._common_public_benchmarks import UCF101TripletDataset


def main() -> None:
    root = Path(os.environ.get("BBNB_UCF101_ROOT", "./data/UCF101"))
    ann_path = Path(os.environ.get("BBNB_UCF101_ANN_PATH", "./data/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist"))
    dataset = UCF101TripletDataset(root=root, annotation_path=ann_path, frames_per_clip=7, train=True, image_size=128)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VFIUNet().to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        last_loss = None
        for prev, mid, nxt in loader:
            prev = prev.to(device)
            mid = mid.to(device)
            nxt = nxt.to(device)
            t = torch.randint(0, cfg.T, (mid.size(0),), device=device)
            loss = lossf(model, mid, (prev, nxt), t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
