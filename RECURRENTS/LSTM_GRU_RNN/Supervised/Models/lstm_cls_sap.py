from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class LSTMSAPConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_classes: int = 2
    pad_idx: int = 0
    bidirectional: bool = False


class SelfAttentivePooling(nn.Module):
    """Single-head self-attentive pooling (SAP) over time.
    w^T tanh(Wx) is equivalent to an additive attention with a learned query.
    """
    def __init__(self, in_dim: int, attn_dim: int):
        super().__init__()
        self.proj = nn.Linear(in_dim, attn_dim)
        self.v = nn.Linear(attn_dim, 1, bias=False)

    def forward(self, seq: torch.Tensor, mask: torch.Tensor):
        # seq: (B, T, D); mask: (B, T) with 1 for valid
        scores = self.v(torch.tanh(self.proj(seq))).squeeze(-1)  # (B, T)
        scores = scores.masked_fill(mask == 0, -1e30)
        weights = torch.softmax(scores, dim=-1)
        ctx = torch.bmm(weights.unsqueeze(1), seq).squeeze(1)  # (B, D)
        return ctx, weights


class LSTMSAPClassifier(nn.Module):
    def __init__(self, cfg: LSTMSAPConfig, attn_dim: int = 128):
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
        feat_dim = cfg.hidden_dim * (2 if cfg.bidirectional else 1)
        self.sap = SelfAttentivePooling(feat_dim, attn_dim)
        self.dropout = nn.Dropout(cfg.dropout)
        self.fc = nn.Linear(feat_dim, cfg.num_classes)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.02)
        for n, p in self.lstm.named_parameters():
            if "weight_" in n:
                nn.init.xavier_uniform_(p)
            elif "bias_" in n:
                nn.init.zeros_(p)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, tokens: torch.LongTensor, lengths: torch.LongTensor):
        emb = self.embedding(tokens)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        seq, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B, T, D)
        B, T, D = seq.shape
        mask = (torch.arange(T, device=seq.device).unsqueeze(0) < lengths.unsqueeze(1)).long()
        ctx, _ = self.sap(seq, mask)
        ctx = self.dropout(ctx)
        return self.fc(ctx)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
