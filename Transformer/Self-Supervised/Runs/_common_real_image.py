from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _folder_exists(path: Path) -> bool:
    return path.is_dir() and any(child.is_dir() for child in path.iterdir())


def _split_indices(num_items: int, val_ratio: float, test_ratio: float, seed: int = 42):
    if num_items < 3:
        raise ValueError("Need at least 3 samples to create train/val/test splits.")
    n_val = max(1, int(num_items * val_ratio))
    n_test = max(1, int(num_items * test_ratio))
    n_train = num_items - n_val - n_test
    if n_train <= 0:
        raise ValueError("Split ratios leave no training samples.")
    order = torch.randperm(num_items, generator=torch.Generator().manual_seed(seed)).tolist()
    train_idx = order[:n_train]
    val_idx = order[n_train:n_train + n_val]
    test_idx = order[n_train + n_val:n_train + n_val + n_test]
    return train_idx, val_idx, test_idx


def _make_base_transforms(image_size: int):
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.4, 0.4, 0.4, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    eval_tf = transforms.Compose([
        transforms.Resize(image_size + 16),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tf, eval_tf


class TwoCropsTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        return self.base_transform(x), self.base_transform(x)


def _make_loaders(train_set, val_set, test_set, batch_size: int, num_workers: int):
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def make_real_image_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    image_size: int = 224,
):
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    train_tf, eval_tf = _make_base_transforms(image_size)

    if _folder_exists(train_dir) and _folder_exists(val_dir) and _folder_exists(test_dir):
        train_set = datasets.ImageFolder(str(train_dir), transform=train_tf)
        val_set = datasets.ImageFolder(str(val_dir), transform=eval_tf)
        test_set = datasets.ImageFolder(str(test_dir), transform=eval_tf)
        return _make_loaders(train_set, val_set, test_set, batch_size, num_workers)

    if _folder_exists(train_dir) and _folder_exists(test_dir):
        full_train = datasets.ImageFolder(str(train_dir), transform=train_tf)
        full_eval = datasets.ImageFolder(str(train_dir), transform=eval_tf)
        n_val = max(1, int(len(full_train) * val_ratio))
        n_train = len(full_train) - n_val
        order = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(42)).tolist()
        train_idx = order[:n_train]
        val_idx = order[n_train:]
        train_set = Subset(full_train, train_idx)
        val_set = Subset(full_eval, val_idx)
        test_set = datasets.ImageFolder(str(test_dir), transform=eval_tf)
        return _make_loaders(train_set, val_set, test_set, batch_size, num_workers)

    if _folder_exists(train_dir):
        full_train = datasets.ImageFolder(str(train_dir), transform=train_tf)
        full_eval = datasets.ImageFolder(str(train_dir), transform=eval_tf)
        train_idx, val_idx, test_idx = _split_indices(len(full_train), val_ratio, test_ratio)
        train_set = Subset(full_train, train_idx)
        val_set = Subset(full_eval, val_idx)
        test_set = Subset(full_eval, test_idx)
        return _make_loaders(train_set, val_set, test_set, batch_size, num_workers)

    if _folder_exists(root):
        full_train = datasets.ImageFolder(str(root), transform=train_tf)
        full_eval = datasets.ImageFolder(str(root), transform=eval_tf)
        train_idx, val_idx, test_idx = _split_indices(len(full_train), val_ratio, test_ratio)
        train_set = Subset(full_train, train_idx)
        val_set = Subset(full_eval, val_idx)
        test_set = Subset(full_eval, test_idx)
        return _make_loaders(train_set, val_set, test_set, batch_size, num_workers)

    raise FileNotFoundError(
        f"No real folder-backed dataset found under {data_root}. "
        "Expected class folders or train/val/test splits."
    )


def make_two_crops_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    num_workers: int = 4,
    image_size: int = 224,
):
    root = Path(data_root)
    train_dir = root / "train"
    val_dir = root / "val"
    test_dir = root / "test"
    train_tf, eval_tf = _make_base_transforms(image_size)
    aug = TwoCropsTransform(train_tf)

    if _folder_exists(train_dir) and _folder_exists(val_dir) and _folder_exists(test_dir):
        train_set = datasets.ImageFolder(str(train_dir), transform=aug)
        val_set = datasets.ImageFolder(str(val_dir), transform=eval_tf)
        test_set = datasets.ImageFolder(str(test_dir), transform=eval_tf)
        return _make_loaders(train_set, val_set, test_set, batch_size, num_workers)

    if _folder_exists(train_dir) and _folder_exists(test_dir):
        full_train = datasets.ImageFolder(str(train_dir), transform=aug)
        full_eval = datasets.ImageFolder(str(train_dir), transform=eval_tf)
        n_val = max(1, int(len(full_train) * val_ratio))
        n_train = len(full_train) - n_val
        order = torch.randperm(len(full_train), generator=torch.Generator().manual_seed(42)).tolist()
        train_idx = order[:n_train]
        val_idx = order[n_train:]
        train_set = Subset(full_train, train_idx)
        val_set = Subset(full_eval, val_idx)
        test_set = datasets.ImageFolder(str(test_dir), transform=eval_tf)
        return _make_loaders(train_set, val_set, test_set, batch_size, num_workers)

    if _folder_exists(train_dir):
        full_train = datasets.ImageFolder(str(train_dir), transform=aug)
        full_eval = datasets.ImageFolder(str(train_dir), transform=eval_tf)
        train_idx, val_idx, test_idx = _split_indices(len(full_train), val_ratio, test_ratio)
        train_set = Subset(full_train, train_idx)
        val_set = Subset(full_eval, val_idx)
        test_set = Subset(full_eval, test_idx)
        return _make_loaders(train_set, val_set, test_set, batch_size, num_workers)

    if _folder_exists(root):
        full_train = datasets.ImageFolder(str(root), transform=aug)
        full_eval = datasets.ImageFolder(str(root), transform=eval_tf)
        train_idx, val_idx, test_idx = _split_indices(len(full_train), val_ratio, test_ratio)
        train_set = Subset(full_train, train_idx)
        val_set = Subset(full_eval, val_idx)
        test_set = Subset(full_eval, test_idx)
        return _make_loaders(train_set, val_set, test_set, batch_size, num_workers)

    raise FileNotFoundError(
        f"No real folder-backed dataset found under {data_root}. "
        "Expected class folders or train/val/test splits."
    )
