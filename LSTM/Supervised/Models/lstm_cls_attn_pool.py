from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class LSTMAttnPoolConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2
    pad_idx: int = 0
    attn_hidden: int = 128  # size of additive attention MLP
    bidirectional: bool = False


class AdditiveAttention(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1)
        )

    def forward(self, seq: torch.Tensor, mask: torch.Tensor):
        # seq: (B, T, D); mask: (B, T) with 1 for valid, 0 for pad
        scores = self.proj(seq).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, -1e30)
        weights = torch.softmax(scores, dim=-1)  # (B, T)
        ctx = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)  # (B, D)
        return ctx, weights


class LSTMAttnPoolClassifier(nn.Module):
    """LSTM + additive attention pooling over time. Many-to-one classification.
    Single attention head; still a single-model supervised setup.
    """
    def __init__(self, cfg: LSTMAttnPoolConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=cfg.bidirectional,
            batch_first=True,
        )
        out_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.attn = AdditiveAttention(out_dim, cfg.attn_hidden)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(out_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for name, p in self.lstm.named_parameters():
            if "weight_" in name:
                nn.init.xavier_uniform_(p)
            elif "bias_" in name:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        seq, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T, D)
        B, T, D = seq.size()
        mask = (torch.arange(T, device=seq.device).unsqueeze(0) < lengths.unsqueeze(1)).long()
        ctx, _ = self.attn(seq, mask)
        ctx = self.dropout(ctx)
        logits = self.fc(ctx)
        return logits

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
