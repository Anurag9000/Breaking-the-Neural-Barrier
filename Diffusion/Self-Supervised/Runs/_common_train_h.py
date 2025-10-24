import os, math
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, utils


def get_cifar_loaders(data_root, batch_size=128, num_workers=4, val_ratio=0.1):
    """
    Prepare CIFAR-10 train, validation, and test DataLoaders.
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

    # Full train dataset
    full = datasets.CIFAR10(root=data_root, train=True, download=True, transform=tf_train)
    val_len = int(len(full) * val_ratio)
    train_len = len(full) - val_len

    train_set, val_set = random_split(
        full, [train_len, val_len], generator=torch.Generator().manual_seed(42)
    )
    # Ensure validation set uses evaluation transforms
    val_set.dataset = datasets.CIFAR10(root=data_root, train=True, download=False, transform=tf_eval)

    test_set = datasets.CIFAR10(root=data_root, train=False, download=True, transform=tf_eval)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)

    return train_loader, val_loader, test_loader


@torch.no_grad()
def random_box_mask(B, H=32, W=32, device='cuda', min_frac=0.25, max_frac=0.5):
    """
    Generate random rectangular masks for inpainting.
    Masked region = 0, unmasked = 1
    """
    mask = torch.ones(B, 1, H, W, device=device)
    for i in range(B):
        fh = torch.empty(1).uniform_(min_frac, max_frac).item()
        fw = torch.empty(1).uniform_(min_frac, max_frac).item()
        hh, ww = int(H * fh), int(W * fw)
        y0 = torch.randint(0, H - hh + 1, (1,)).item()
        x0 = torch.randint(0, W - ww + 1, (1,)).item()
        mask[i, :, y0:y0+hh, x0:x0+ww] = 0.0
    return mask


@torch.no_grad()
def to_gray(x):
    """Convert RGB images to grayscale (batch-wise)."""
    r, g, b = x[:, 0:1], x[:, 1:2], x[:, 2:3]
    return 0.2989*r + 0.5870*g + 0.1140*b


@torch.no_grad()
def make_lr(x, scale=4):
    """Create low-resolution version of an image and upsample back to original size."""
    x_lr = F.interpolate(x, scale_factor=1.0/scale, mode='area')
    return F.interpolate(x_lr, scale_factor=scale, mode='nearest')


class EarlyStopper:
    """Simple early stopping utility based on validation loss."""
    def __init__(self, patience=20, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.best = math.inf
        self.counter = 0

    def step(self, val):
        """Update early stopper with new validation metric."""
        if val < self.best - self.min_delta:
            self.best = val
            self.counter = 0
            return True
        self.counter += 1
        return False

    def should_stop(self):
        return self.counter >= self.patience


@torch.no_grad()
def save_samples(grid, path):
    """Save a batch of images as a grid to the specified path."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    utils.save_image(grid, path)
