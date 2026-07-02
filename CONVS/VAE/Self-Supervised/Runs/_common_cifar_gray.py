from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


_DEF_TFM = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])


def make_cifar10_gray_loaders(root: str | Path, batch_size: int, val_split: int = 5000, num_workers: int = 0) -> Tuple[DataLoader, DataLoader, DataLoader]:
    train = datasets.CIFAR10(root=root, train=True, download=True, transform=_DEF_TFM)
    test = datasets.CIFAR10(root=root, train=False, download=True, transform=_DEF_TFM)
    tr_len = len(train) - int(val_split)
    tr, va = random_split(train, [tr_len, int(val_split)], generator=torch.Generator().manual_seed(42))
    common = dict(batch_size=int(batch_size), num_workers=int(num_workers), pin_memory=torch.cuda.is_available())
    return (
        DataLoader(tr, shuffle=True, **common),
        DataLoader(va, shuffle=False, **common),
        DataLoader(test, shuffle=False, **common),
    )
