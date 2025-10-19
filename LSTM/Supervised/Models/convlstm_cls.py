from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class ConvLSTMConfig:
    in_channels: int = 1     # e.g., grayscale frames
    hidden_channels: int = 32
    kernel_size: int = 3
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 10


class ConvLSTMCell(nn.Module):
    def __init__(self, in_ch, hid_ch, k=3, padding=1):
        super().__init__()
        self.hid_ch = hid_ch
        self.conv = nn.Conv2d(in_ch + hid_ch, 4*hid_ch, kernel_size=k, padding=padding)

    def forward(self, x, state):
        # x: (B, C, H, W)
        h, c = state
        inp = torch.cat([x, h], dim=1)
        gates = self.conv(inp)
        i, f, g, o = gates.chunk(4, dim=1)
        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        c_next = f * c + i * g
        o = torch.sigmoid(o)
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMClassifier(nn.Module):
    """ConvLSTM over frame sequences -> GAP -> linear head (many-to-one)."""
    def __init__(self, cfg: ConvLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.layers = nn.ModuleList()
        for l in range(cfg.num_layers):
            in_ch = cfg.in_channels if l == 0 else cfg.hidden_channels
            self.layers.append(ConvLSTMCell(in_ch, cfg.hidden_channels, k=cfg.kernel_size, padding=cfg.kernel_size//2))
        self.dropout = nn.Dropout(cfg.dropout)
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(cfg.hidden_channels, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu');
                if m.bias is not None: nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, frames: torch.Tensor, lengths: torch.Tensor):
        # frames: (B, T, C, H, W); lengths: (B,)
        B, T, C, H, W = frames.shape
        h = [frames.new_zeros(B, self.cfg.hidden_channels, H, W) for _ in self.layers]
        c = [frames.new_zeros(B, self.cfg.hidden_channels, H, W) for _ in self.layers]
        last_h = None
        for t in range(T):
            x = frames[:, t]
            for l, cell in enumerate(self.layers):
                h_l, c_l = cell(x, (h[l], c[l]))
                mask = (t < lengths).float().view(B, 1, 1, 1)
                h[l] = h_l * mask + h[l] * (1 - mask)
                c[l] = c_l * mask + c[l] * (1 - mask)
                x = h[l]
            last_h = h[-1]
        feat = self.gap(last_h).view(B, -1)
        feat = self.dropout(feat)
        return self.fc(feat)

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
