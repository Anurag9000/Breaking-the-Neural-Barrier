import torch
import torch.nn as nn
import torch.nn.functional as F


class MSEEnergy(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, target):
        return ((x - target) ** 2).mean()


class TotalVariation(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.w = weight

    def forward(self, x):
        dx = (x[:, :, 1:, :] - x[:, :, :-1, :]).abs().mean()
        dy = (x[:, :, :, 1:] - x[:, :, :, :-1]).abs().mean()
        return self.w * (dx + dy)


class SimpleClassifier(nn.Module):
    """
    Small conv classifier to produce an energy −log p(y|x).
    This classifier can be trained jointly with the diffusion net.
    """
    def __init__(self, num_classes=10):
        super().__init__()
        self.f = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AvgPool2d(2),
            nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1)
        )
        self.head = nn.Linear(256, num_classes)

    def forward(self, x):
        h = self.f(x).flatten(1)
        return self.head(h)

    def energy(self, x, y):
        logits = self(x)
        return F.cross_entropy(logits, y)
