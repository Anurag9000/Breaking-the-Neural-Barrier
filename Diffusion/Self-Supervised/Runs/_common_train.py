# runs/_common_train.py
import os
import math
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils


# ----------------------------------------------------------
# CIFAR-10 Dataloaders
# ----------------------------------------------------------
def make_cifar10_loaders(
    data_root: str = "./data",
    batch_size: int = 128,
    val_ratio: float = 0.1,
    num_workers: int = 4,
):
    """Create train, validation, and test dataloaders for CIFAR-10."""

    # Data augmentation and normalization
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    # Load full training set
    full = datasets.CIFAR10(
        root=data_root, train=True, download=True, transform=tf_train
    )

    # Split into train and validation sets
    val_len = int(len(full) * val_ratio)
    train_len = len(full) - val_len
    train_set, val_set = random_split(
        full, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    # Ensure eval transforms are used for validation and test
    val_set.dataset = datasets.CIFAR10(
        root=data_root, train=True, download=False, transform=tf_eval
    )
    test_set = datasets.CIFAR10(
        root=data_root, train=False, download=True, transform=tf_eval
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# ----------------------------------------------------------
# Early Stopping Utility
# ----------------------------------------------------------
class EarlyStopper:
    """Simple early stopping utility based on validation loss."""

    def __init__(self, patience: int = 20, min_delta: float = 0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.counter = 0

    def step(self, val: float) -> bool:
        """Return True if improvement, else False."""
        if val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
            return True
        else:
            self.counter += 1
            return False

    def should_stop(self) -> bool:
        """Return True if patience has been exceeded."""
        return self.counter >= self.patience


# ----------------------------------------------------------
# Image Saving Utility
# ----------------------------------------------------------
@torch.no_grad()
def save_samples(grid: torch.Tensor, path: str):
    """Save a grid of images to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)
