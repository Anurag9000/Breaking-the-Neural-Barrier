from __future__ import annotations

from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


def _image_to_sequence(x: torch.Tensor) -> torch.Tensor:
    # x: (C, H, W) -> (H, C*W)
    return x.permute(1, 2, 0).reshape(x.size(1), -1)


class Caltech256SequenceDataset(Dataset):
    def __init__(self, root: str | Path, image_size: int = 64):
        self.dataset = datasets.Caltech256(
            root=str(root),
            download=True,
            transform=transforms.Compose(
                [
                    transforms.Resize((image_size, image_size)),
                    transforms.ToTensor(),
                ]
            ),
        )
        self.num_classes = 257
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        x, y = self.dataset[index]
        return _image_to_sequence(x), y


def make_caltech256_loaders(
    root: str | Path,
    batch_size: int = 128,
    image_size: int = 64,
    num_workers: int = 0,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, int, int]:
    dataset = Caltech256SequenceDataset(root=root, image_size=image_size)
    n = len(dataset)
    n_train = int(n * 0.8)
    n_val = int(n * 0.1)
    n_test = n - n_train - n_val
    train_set, val_set, test_set = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    def collate(batch):
        xs, ys = zip(*batch)
        return torch.stack(list(xs), 0), torch.tensor(ys, dtype=torch.long)

    loader_kwargs = dict(batch_size=batch_size, num_workers=num_workers, collate_fn=collate, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(train_set, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_set, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_set, shuffle=False, **loader_kwargs)
    return train_loader, val_loader, test_loader, 192, dataset.num_classes

