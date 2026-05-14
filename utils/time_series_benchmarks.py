from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import torch
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset


class FordADataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        x = np.asarray(features, dtype=np.float32)
        x = x[:, None, :]
        x = x - x.mean(axis=2, keepdims=True)
        std = x.std(axis=2, keepdims=True)
        std[std < 1e-6] = 1.0
        x = x / std
        self.x = torch.from_numpy(x)
        self.y = torch.from_numpy(labels.astype(np.int64))

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


class FordASequenceDataset(Dataset):
    def __init__(self, features: np.ndarray):
        x = np.asarray(features, dtype=np.float32)
        x = x[:, None, :]
        x = x - x.mean(axis=2, keepdims=True)
        std = x.std(axis=2, keepdims=True)
        std[std < 1e-6] = 1.0
        x = x / std
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx]


class FordAIndexedDataset(Dataset):
    def __init__(self, features: np.ndarray):
        x = np.asarray(features, dtype=np.float32)
        x = x[:, None, :]
        x = x - x.mean(axis=2, keepdims=True)
        std = x.std(axis=2, keepdims=True)
        std[std < 1e-6] = 1.0
        x = x / std
        self.x = torch.from_numpy(x)

    def __len__(self) -> int:
        return self.x.size(0)

    def __getitem__(self, idx: int):
        return self.x[idx], torch.tensor(idx, dtype=torch.long)


def _load_forda_arrays():
    data = fetch_openml(name="FordA", version=1, as_frame=False)
    x = np.asarray(data.data, dtype=np.float32)
    y = np.asarray(data.target)
    y = np.where(y.astype(str) == "-1", 0, 1).astype(np.int64)
    return x, y


def make_forda_loaders(
    *,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 0,
    num_workers: int = 0,
):
    x, y = _load_forda_arrays()
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_fraction, random_state=seed, stratify=y)
    val_size = max(1, int(len(x_train) * val_fraction))
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=val_size, random_state=seed, stratify=y_train)

    train_ds = FordADataset(x_train, y_train)
    val_ds = FordADataset(x_val, y_val)
    test_ds = FordADataset(x_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    return train_loader, val_loader, test_loader, 2


def make_forda_sequence_loaders(
    *,
    batch_size: int = 128,
    val_fraction: float = 0.1,
    test_fraction: float = 0.2,
    seed: int = 0,
    num_workers: int = 0,
    return_index: bool = False,
):
    x, y = _load_forda_arrays()
    x_train, x_test, _, _ = train_test_split(x, y, test_size=test_fraction, random_state=seed, stratify=y)
    val_size = max(1, int(len(x_train) * val_fraction))
    x_train, x_val = train_test_split(x_train, test_size=val_size, random_state=seed)

    train_ds = FordAIndexedDataset(x_train) if return_index else FordASequenceDataset(x_train)
    val_ds = FordAIndexedDataset(x_val) if return_index else FordASequenceDataset(x_val)
    test_ds = FordAIndexedDataset(x_test) if return_index else FordASequenceDataset(x_test)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return train_loader, val_loader, test_loader, 1
