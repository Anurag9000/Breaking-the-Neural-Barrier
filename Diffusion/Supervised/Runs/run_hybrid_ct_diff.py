from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.hybrid_ct_diff import HybridCTDiff
from runs._common_public_benchmarks import PairedImageFolderDataset


def main() -> None:
    source_root = Path(os.environ.get("BBNB_CT_SOURCE_ROOT", "./data/ct_source"))
    target_root = Path(os.environ.get("BBNB_CT_TARGET_ROOT", "./data/ct_target"))
    dataset = PairedImageFolderDataset(source_root=source_root, target_root=target_root, image_size=128)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=0, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = HybridCTDiff().to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=2e-4)

    for epoch in range(3):
        last_loss = None
        for x, _target in loader:
            x = x.to(device)
            t = torch.randint(0, cfg.T, (x.size(0),), device=device)
            loss = lossf(model, x, None, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
