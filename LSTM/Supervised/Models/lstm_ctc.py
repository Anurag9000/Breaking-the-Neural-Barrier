from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class LSTMCTCConfig:
    vocab_size: int = 64   # input token vocab (for embedding)
    emb_dim: int = 64
    hidden_dim: int = 128
    num_layers: int = 2
    dropout: float = 0.1
    num_labels: int = 20   # label alphabet size (without blank)
    blank: int = 0         # CTC blank index in [0..num_labels]
    pad_idx: int = 0


class LSTMCTC(nn.Module):
    """Many-to-many unaligned seq labeling with CTC loss.
    Emits per-timestep logits over (num_labels+1) classes including blank.
    """
    def __init__(self, cfg: LSTMCTCConfig):
        super().__init__()
        self.cfg = cfg
        self.embedding = nn.Embedding(cfg.vocab_size, cfg.emb_dim, padding_idx=cfg.pad_idx)
        self.lstm = nn.LSTM(
            input_size=cfg.emb_dim,
            hidden_size=cfg.hidden_dim,
            num_layers=cfg.num_layers,
            dropout=cfg.dropout if cfg.num_layers > 1 else 0.0,
            bidirectional=True,
            batch_first=True,
        )
        out_dim = cfg.hidden_dim * 2
        self.emitter = nn.Linear(out_dim, cfg.num_labels + 1)  # +1 for blank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for n, p in self.lstm.named_parameters():
            if 'weight_' in n:
                nn.init.xavier_uniform_(p)
            elif 'bias_' in n:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.emitter.weight)
        nn.init.zeros_(self.emitter.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        seq, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B,T,2H)
        logits = self.emitter(seq)  # (B,T,C)
        # CTC expects (T,B,C)
        return logits.transpose(0, 1)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
