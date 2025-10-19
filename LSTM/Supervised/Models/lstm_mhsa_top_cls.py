from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class MHSAonLSTMConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_heads: int = 4
    num_classes: int = 2
    pad_idx: int = 0


class LSTMMHSAClassifier(nn.Module):
    """LSTM encoder + Multi-Head Self-Attention top; pool with masked mean."""
    def __init__(self, cfg: MHSAonLSTMConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=False,
            batch_first=True,
        )
        self.mhsa = nn.MultiheadAttention(embed_dim=cfg.hidden_dim, num_heads=cfg.num_heads, batch_first=True)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(cfg.hidden_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for n, p in self.lstm.named_parameters():
            if 'weight_' in n: nn.init.xavier_uniform_(p)
            elif 'bias_' in n: nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)  # (B,T,E)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        seq, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B,T,H)
        B, T, H = seq.shape
        # key_padding_mask: (B,T) where True=pad positions
        kpm = torch.arange(T, device=seq.device).unsqueeze(0) >= lengths.unsqueeze(1)
        attn_out, _ = self.mhsa(seq, seq, seq, key_padding_mask=kpm)
        # masked mean pooling
        mask = (~kpm).float().unsqueeze(-1)
        sum_feat = (attn_out * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        feat = sum_feat / denom
        return self.fc(self.dropout(feat))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
