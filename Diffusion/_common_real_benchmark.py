from __future__ import annotations

from typing import Tuple

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def make_real_diffusion_smoke_loaders(
    dataset_name: str = 'CIFAR10',
    batch_size: int = 8,
    *,
    root: str = './data',
    num_workers: int = 0,
    val_split: float = 0.1,
) -> Tuple[DataLoader, DataLoader, DataLoader, int]:
    if dataset_name.upper() not in {'CIFAR10', 'CIFAR100'}:
        raise ValueError(f'Unsupported diffusion smoke dataset: {dataset_name}')
    tf = transforms.Compose([transforms.ToTensor()])
    ds_cls = datasets.CIFAR10 if dataset_name.upper() == 'CIFAR10' else datasets.CIFAR100
    full = ds_cls(root=root, train=True, transform=tf, download=True)
    n = len(full)
    n_val = max(1, int(n * val_split))
    n_train = max(1, n - n_val)
    train_ds, val_ds = random_split(full, [n_train, n_val], generator=None)
    test_ds = ds_cls(root=root, train=False, transform=tf, download=True)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    dl_test = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return dl_train, dl_val, dl_test, 3
