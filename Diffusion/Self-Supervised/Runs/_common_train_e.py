# runs/_common_train_e.py
import os
import math
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils


def make_cifar10_loaders(data_root='./data', batch_size=128, val_ratio=0.1, num_workers=4):
    """
    Creates CIFAR-10 train, validation, and test DataLoaders.

    Args:
        data_root (str): Directory to store/download the dataset.
        batch_size (int): Batch size for loaders.
        val_ratio (float): Fraction of training set to use as validation.
        num_workers (int): Number of DataLoader workers.

    Returns:
        train_loader, val_loader, test_loader
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

    full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    val_len = int(len(full) * val_ratio)
    train_len = len(full) - val_len
    train_set, val_set = random_split(
        full, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    val_set.dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tf_eval)
    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_eval)

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


class EarlyStopper:
    """
    Early stopping utility.
    """
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.counter = 0

    def step(self, val):
        """
        Update counter based on new validation value.

        Returns True if this is the new best value.
        """
        if val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self):
        """
        Returns True if early stopping criterion is met.
        """
        return self.counter >= self.patience


@torch.no_grad()
def save_samples(grid, path):
    """
    Save a grid of images to a file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)
