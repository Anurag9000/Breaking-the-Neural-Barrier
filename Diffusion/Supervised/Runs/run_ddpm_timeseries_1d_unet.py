from __future__ import annotations

import os
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from core.diffusion_core import BetaSchedule, DiffusionConfig, DiffusionLoss
from models.ddpm_timeseries_1d_unet import TS1DUNet
from runs._common_public_benchmarks import LibriSpeechWaveformDataset


def main() -> None:
    root = Path(os.environ.get("BBNB_LIBRISPEECH_ROOT", "./data/LibriSpeech"))
    dataset = LibriSpeechWaveformDataset(root=root, url=os.environ.get("BBNB_LIBRISPEECH_URL", "train-clean-100"), segment_seconds=1.0, sample_rate=16000, train=True)
    loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=torch.cuda.is_available())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = TS1DUNet().to(device)
    cfg = DiffusionConfig(T=1000, objective="eps")
    sched = BetaSchedule(cfg.T, cfg.beta_start, cfg.beta_end)
    lossf = DiffusionLoss(cfg, sched)
    opt = optim.AdamW(model.parameters(), lr=1e-4)

    for epoch in range(3):
        last_loss = None
        for waveform, _transcript in loader:
            waveform = waveform.to(device)
            t = torch.randint(0, cfg.T, (waveform.size(0),), device=device)
            loss = lossf(model, waveform, None, t)
            opt.zero_grad()
            loss.backward()
            opt.step()
            last_loss = loss.item()
        print(f"epoch {epoch}: loss {last_loss:.4f}")


if __name__ == "__main__":
    main()
