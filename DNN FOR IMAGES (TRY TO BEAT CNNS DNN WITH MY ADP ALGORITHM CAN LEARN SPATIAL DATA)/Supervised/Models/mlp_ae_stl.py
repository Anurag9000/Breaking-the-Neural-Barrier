
import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int, use_bn: bool=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.bn = nn.BatchNorm1d(out_features) if use_bn else None
        self.act = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.linear(x)
        if self.bn is not None:
            x = self.bn(x)
        return self.act(x)

class MLPAutoencoder(nn.Module):
    """
    Symmetric MLP Autoencoder for images (no CNNs).
    - Encoder hidden widths: e.g., [512, 256]
    - Bottleneck width: single int
    - Decoder mirrors encoder in reverse back to input dimension.
    """
    def __init__(self, in_dim: int, hidden_widths, bottleneck: int, use_bn: bool=True, output_activation: str="sigmoid"):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_widths = list(hidden_widths)
        self.bottleneck = int(bottleneck)
        self.use_bn = use_bn
        self.output_activation = output_activation

        # Build encoder
        enc_layers = []
        prev = in_dim
        for w in self.hidden_widths:
            enc_layers.append(MLPBlock(prev, w, use_bn))
            prev = w
        self.enc = nn.Sequential(*enc_layers)

        self.fc_mu = nn.Linear(prev, self.bottleneck)

        # Build decoder (mirror)
        dec_layers = []
        prev = self.bottleneck
        for w in reversed(self.hidden_widths):
            dec_layers.append(MLPBlock(prev, w, use_bn))
            prev = w
        self.dec = nn.Sequential(*dec_layers)
        self.out = nn.Linear(prev, in_dim)

    def encode(self, x):
        return self.fc_mu(self.enc(x))

    def decode(self, z):
        x = self.dec(z)
        x = self.out(x)
        if self.output_activation == "sigmoid":
            x = torch.sigmoid(x)
        elif self.output_activation == "tanh":
            x = torch.tanh(x)
        return x

    def forward(self, img):
        # img: (B, C, H, W) -> flatten
        x = img.view(img.size(0), -1)
        z = self.encode(x)
        xr = self.decode(z)
        return xr
