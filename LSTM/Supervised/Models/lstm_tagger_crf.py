from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


@dataclass
class LSTMCRFConfig:
    vocab_size: int = 20000
    emb_dim: int = 128
    hidden_dim: int = 256
    num_layers: int = 1
    dropout: float = 0.1
    num_tags: int = 10
    pad_idx: int = 0


class LinearCRF(nn.Module):
    """Simple linear-chain CRF with start/end transitions.
    Expects emissions of shape (B, T, K) and valid lengths.
    """
    def __init__(self, num_tags: int):
        super().__init__()
        self.K = num_tags
        self.start = nn.Parameter(torch.zeros(num_tags))
        self.end = nn.Parameter(torch.zeros(num_tags))
        self.transitions = nn.Parameter(torch.zeros(num_tags, num_tags))  # from i -> j

    def _logsumexp(self, x: torch.Tensor, dim=-1):
        m, _ = x.max(dim=dim, keepdim=True)
        return m + (x - m).exp().sum(dim=dim, keepdim=True).log()

    def neg_log_likelihood(self, emissions: torch.Tensor, tags: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        # emissions: (B,T,K); tags: (B,T) with -100 for pads; lengths: (B,)
        B, T, K = emissions.shape
        # Compute log-partition Z via forward algorithm
        log_alpha = self.start + emissions[:, 0]  # (B,K)
        for t in range(1, T):
            emit_t = emissions[:, t].unsqueeze(2)  # (B,K,1)
            trans = self.transitions.unsqueeze(0)  # (1,K,K)
            score = log_alpha.unsqueeze(1) + trans + emit_t  # (B,K,K)
            log_alpha = self._logsumexp(score, dim=2).squeeze(2)  # (B,K)
        log_Z = self._logsumexp(log_alpha + self.end, dim=1).squeeze(1)  # (B,)

        # Compute score of the given tag path
        path_score = self.start[tags[:, 0].clamp(min=0)]
        path_score += emissions[torch.arange(B), 0, tags[:, 0].clamp(min=0)]
        for t in range(1, T):
            curr = tags[:, t].clamp(min=0)
            prev = tags[:, t - 1].clamp(min=0)
            trans_score = self.transitions[prev, curr]
            emit_score = emissions[torch.arange(B), t, curr]
            # Zero out padded positions
            mask_t = (t < lengths).float()
            path_score += trans_score * mask_t + emit_score * mask_t
        path_score += self.end[tags[torch.arange(B), lengths - 1]]
        # Mask batches to valid lengths
        return (log_Z - path_score).mean()

    def decode(self, emissions: torch.Tensor, lengths: torch.Tensor):
        # Viterbi
        B, T, K = emissions.shape
        delta = self.start + emissions[:, 0]   # (B,K)
        psi = emissions.new_zeros((B, T, K), dtype=torch.long)
        for t in range(1, T):
            score = delta.unsqueeze(1) + self.transitions.unsqueeze(0)  # (B,K,K)
            best_val, best_idx = score.max(dim=2)  # (B,K)
            delta = best_val + emissions[:, t]
            psi[:, t] = best_idx
        delta = delta + self.end
        best_last = delta.argmax(dim=1)  # (B,)
        paths = []
        for b in range(B):
            L = int(lengths[b].item())
            last = best_last[b].item()
            path = [last]
            for t in range(L - 1, 0, -1):
                last = psi[b, t, last].item()
                path.append(last)
            path.reverse()
            paths.append(path)
        return paths


class LSTMCRFTagger(nn.Module):
    def __init__(self, cfg: LSTMCRFConfig):
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
        self.dropout = nn.Dropout(cfg.dropout)
        self.emitter = nn.Linear(cfg.hidden_dim, cfg.num_tags)
        self.crf = LinearCRF(cfg.num_tags)
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
        seq, _ = pad_packed_sequence(packed_out, batch_first=True)  # (B,T,H)
        seq = self.dropout(seq)
        emissions = self.emitter(seq)  # (B,T,K)
        return emissions

    def loss(self, emissions, tags, lengths):
        return self.crf.neg_log_likelihood(emissions, tags, lengths)

    def decode(self, emissions, lengths):
        return self.crf.decode(emissions, lengths)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
