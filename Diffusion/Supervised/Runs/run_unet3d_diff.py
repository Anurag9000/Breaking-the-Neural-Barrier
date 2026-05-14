from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.unet3d_diff import UNet3dDiff
from runs._common_public_benchmarks import UCF101ClipDataset


def main() -> None:
    root = Path(os.environ.get("BBNB_UCF101_ROOT", "./data/UCF101"))
    ann_path = Path(os.environ.get("BBNB_UCF101_ANN_PATH", "./data/UCF101/UCF101TrainTestSplits-RecognitionTask/ucfTrainTestlist"))
    dataset = UCF101ClipDataset(root=root, annotation_path=ann_path, frames_per_clip=16, train=True, image_size=64)
    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = UNet3dDiff(in_ch=3, out_ch=3).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        last_loss = None
        for video in loader:
            video = video.to(device)
            t = torch.randint(0, cfg.T, (video.size(0),), device=device)
            loss = lossf(model, video, None, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
