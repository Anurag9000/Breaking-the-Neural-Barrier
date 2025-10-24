# runs/_common_train_c.py
import os
import math
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils


# -------------------------------------------------------
# CIFAR-10 Data Loaders
# -------------------------------------------------------
def make_cifar10_loaders(data_root='./data', batch_size=128, val_ratio=0.1, num_workers=4):
    """
    Creates train, validation, and test DataLoaders for CIFAR-10.
    Includes simple data augmentation for training.
    """
    tf_train = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    tf_eval = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # Load and split dataset
    full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    val_len = int(len(full) * val_ratio)
    train_len = len(full) - val_len

    train_set, val_set = random_split(
        full, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )

    # Replace transform for validation subset (no augmentation)
    val_set.dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tf_eval)

    # Test set
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_eval)

    # DataLoaders
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


# -------------------------------------------------------
# Early Stopping Utility
# -------------------------------------------------------
class EarlyStopper:
    def __init__(self, patience=20, min_delta=0.0):
        """
        Early stopping helper.
        Stops training when validation loss stops improving.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.counter = 0

    def step(self, val):
        """Updates early stopping state with new validation loss."""
        if val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self):
        """Checks if training should stop early."""
        return self.counter >= self.patience


# -------------------------------------------------------
# Utility: Save image grid to file
# -------------------------------------------------------
@torch.no_grad()
def save_samples(grid, path):
    """
    Saves a grid of sample images to disk.
    Automatically creates directories if needed.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)
