from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class TPPConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2
    pad_idx: int = 0
    pyramid_levels: tuple = (1, 2, 4)


class LSTMTPPClassifier(nn.Module):
    """LSTM + Temporal Pyramid Pooling (TPP) over hidden states: concat pooled segments."""
    def __init__(self, cfg: TPPConfig):
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
        feat_dim = cfg.hidden_dim * sum(cfg.pyramid_levels)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(feat_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for n, p in self.lstm.named_parameters():
            if 'weight_' in n: nn.init.xavier_uniform_(p)
            elif 'bias_' in n: nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight); nn.init.zeros_(self.fc.bias)

    def _tpp(self, seq: torch.Tensor, lengths: torch.Tensor):
        # seq: (B,T,H); lengths: (B,)
        B, T, H = seq.shape
        feats = []
        for L in self.cfg.pyramid_levels:
            # split valid region into L segments and mean-pool with masking
            for i in range(L):
                # segment [i/L, (i+1)/L) over valid length
                bsz = []
                for b in range(B):
                    Lb = int(lengths[b].item())
                    s = int(i * Lb / L)
                    e = max(s+1, int((i+1) * Lb / L))
                    seg = seq[b, s:e]
                    bsz.append(seg.mean(dim=0))
                feats.append(torch.stack(bsz, dim=0))
        return torch.cat(feats, dim=-1)  # (B, H * sum(levels))

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        seq, _ = pad_packed_sequence(packed_out, batch_first=True)
        pooled = self._tpp(seq, lengths)
        return self.fc(self.dropout(pooled))

    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
