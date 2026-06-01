from __future__ import annotations

import os
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_ucf101_loaders(
    root: str | Path,
    annotation_path: str | Path,
    batch_size: int = 32,
    frames_per_clip: int = 16,
    step_between_clips: int = 8,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    transform = transforms.Compose(
        [
            transforms.ConvertImageDtype(torch.float32),
            transforms.Resize((128, 128)),
        ]
    )
    train_ds = datasets.UCF101(
        root=str(root),
        annotation_path=str(annotation_path),
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        fold=1,
        train=True,
        transform=transform,
        output_format="TCHW",
    )
    test_ds = datasets.UCF101(
        root=str(root),
        annotation_path=str(annotation_path),
        frames_per_clip=frames_per_clip,
        step_between_clips=step_between_clips,
        fold=1,
        train=False,
        transform=transform,
        output_format="TCHW",
    )
    n_val = max(1, int(len(train_ds) * 0.1))
    n_train = len(train_ds) - n_val
    train_ds, val_ds = random_split(train_ds, [n_train, n_val], generator=torch.Generator().manual_seed(seed))

    def collate(batch):
        videos, _audios, labels = zip(*batch)
        lengths = torch.tensor([v.size(0) for v in videos], dtype=torch.long)
        return torch.stack(list(videos), 0), lengths, torch.tensor(labels, dtype=torch.long)

    kwargs = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_ds, shuffle=True, **kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, **kwargs)
    return train_loader, val_loader, test_loader, frames_per_clip, len(train_ds.dataset.classes)
