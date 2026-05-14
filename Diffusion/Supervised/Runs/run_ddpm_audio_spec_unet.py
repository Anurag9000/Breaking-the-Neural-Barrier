from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.ddpm_audio_spec_unet import AudioSpecUNet
from runs._common_public_benchmarks import LibriSpeechSpecDataset


def main() -> None:
    root = Path(os.environ.get("BBNB_LIBRISPEECH_ROOT", "./data/LibriSpeech"))
    dataset = LibriSpeechSpecDataset(root=root, url=os.environ.get("BBNB_LIBRISPEECH_URL", "train-clean-100"), segment_seconds=3.0, sample_rate=16000, n_mels=128, train=True)
    loader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioSpecUNet().to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(3):
        last_loss = None
        for spec, _transcript in loader:
            spec = spec.to(device)
            t = torch.randint(0, cfg.T, (spec.size(0),), device=device)
            loss = lossf(model, spec, None, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
