import torch
import torch.nn as nn


class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bn: bool = True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)


class MLPClassifier(nn.Module):
    """
    Fully-connected (non-CNN) image classifier.
    Flattens image (B,C,H,W) -> (B, C*H*W) then predicts logits.
    """

    def __init__(self, in_dim: int, hidden_widths, num_classes: int = 10, use_bn: bool = True):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden_widths = list(hidden_widths)
        self.num_classes = int(num_classes)
        self.use_bn = bool(use_bn)

        layers = []
        prev = self.in_dim
        for w in self.hidden_widths:
            layers.append(MLPBlock(prev, int(w), use_bn))
            prev = int(w)
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(prev, self.num_classes)

    def forward(self, img):
        x = img.view(img.size(0), -1)
        h = self.backbone(x)
        return self.head(h)

