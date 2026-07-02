from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.ddpm_textcond_unet import TextCondUNet
from runs._common_public_benchmarks import CocoCaptionDataset


def main() -> None:
    root = Path(os.environ.get("BBNB_COCO_ROOT", "./data/coco"))
    ann_file = Path(os.environ.get("BBNB_COCO_TRAIN_ANN", root / "annotations" / "captions_train2017.json"))
    dataset = CocoCaptionDataset(root=root, ann_file=ann_file, image_size=224, text_dim=512)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TextCondUNet(text_dim=512).to(device)
    cfg = DiffusionConfig(T=1000, objective="eps", p_uncond=0.1)
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        last_loss = None
        for x, txt in loader:
            x = x.to(device)
            txt = txt.to(device)
            t = torch.randint(0, cfg.T, (x.size(0),), device=device)
            if torch.rand(()) < cfg.p_uncond:
                txt = torch.zeros_like(txt)
            loss = lossf(model, x, txt, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
