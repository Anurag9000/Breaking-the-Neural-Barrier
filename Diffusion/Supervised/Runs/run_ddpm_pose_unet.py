from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.ddpm_pose_unet import PoseUNet
from runs._common_public_benchmarks import CocoKeypointDataset


def main() -> None:
    root = Path(os.environ.get("BBNB_COCO_ROOT", "./data/coco"))
    ann_file = Path(os.environ.get("BBNB_COCO_KEYPOINTS_ANN", root / "annotations" / "person_keypoints_train2017.json"))
    dataset = CocoKeypointDataset(root=root, ann_file=ann_file, image_size=128)
    loader = DataLoader(dataset, batch_size=16, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = PoseUNet(joints=17).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        last_loss = None
        for img, hm in loader:
            img = img.to(device)
            hm = hm.to(device)
            t = torch.randint(0, cfg.T, (img.size(0),), device=device)
            loss = lossf(model, hm, img, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
