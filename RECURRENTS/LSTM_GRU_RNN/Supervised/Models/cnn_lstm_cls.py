from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class CNNLSTMConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    conv_channels: int = 128
    conv_kernel: int = 5
    conv_stride: int = 1
    conv_padding: int = 2
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2
    pad_idx: int = 0


class CNNLSTMClassifier(nn.Module):
    """1D CNN front-end for local patterns -> LSTM for long context -> CLS.
    """
    def __init__(self, cfg: CNNLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.conv = nn.Conv1d(cfg.emb_dim, cfg.conv_channels, kernel_size=cfg.conv_kernel,
                              stride=cfg.conv_stride, padding=cfg.conv_padding)
        self.act = nn.ReLU(inplace=True)
        self.lstm = nn.LSTM(
            input_size=cfg.conv_channels,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        nn.init.kaiming_uniform_(self.conv.weight, nonlinearity='relu')
        nn.init.zeros_(self.conv.bias)
        for n, p in self.lstm.named_parameters():
            if 'weight_' in n:
                nn.init.xavier_uniform_(p)
            elif 'bias_' in n:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)            # (B, T, E)
        x = emb.transpose(1, 2)                 # (B, E, T)
        x = self.act(self.conv(x))              # (B, C, T)
        x = x.transpose(1, 2)                   # (B, T, C)
        # updated lengths are the same since stride=1 and padding keeps length
        packed = pack_padded_sequence(x, lengths.cpu(), batch_first=True, enforce_sorted=False)
        _, (h_n, _) = self.lstm(packed)
        feat = self.dropout(h_n[-1])
        return self.fc(feat)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
