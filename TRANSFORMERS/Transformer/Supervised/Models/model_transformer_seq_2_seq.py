import math
from typing import Optional, Tuple
import torch
import torch.nn as nn

# --- Positional Encoding ---
class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, d_model)
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


# --- Token Embedding ---
class TokenEmbedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        nn.init.normal_(self.emb.weight, mean=0.0, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.emb(tokens)


class TransformerSeq2Seq(nn.Module):
    """
    A clean encoder-decoder Transformer (Vaswani et al. 2017) suitable for supervised
    text-to-text tasks (translation, summarization-as-supervised, etc.). Single-model only.
    """
    def __init__(
        self,
        src_vocab: int,
        tgt_vocab: int,
        d_model: int = 256,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_len: int = 1024,
    ):
        super().__init__()
        self.src_tok = TokenEmbedding(src_vocab, d_model)
        self.tgt_tok = TokenEmbedding(tgt_vocab, d_model)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.generator = nn.Linear(d_model, tgt_vocab)
        nn.init.xavier_uniform_(self.generator.weight)
        nn.init.zeros_(self.generator.bias)

    def _generate_square_subsequent_mask(self, sz: int, device) -> torch.Tensor:
        mask = torch.full((sz, sz), float('-inf'), device=device)
        mask = torch.triu(mask, diagonal=1)
        return mask

    def forward(
        self,
        src_tokens: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor],
        tgt_tokens_in: torch.Tensor,
        tgt_key_padding_mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        # src/tgt tokens: (B, S), (B, T)
        src = self.pos(self.src_tok(src_tokens))
        tgt = self.pos(self.tgt_tok(tgt_tokens_in))
        tgt_mask = self._generate_square_subsequent_mask(tgt.size(1), tgt.device)
        memory = self.transformer.encoder(
            src,
            src_key_padding_mask=src_key_padding_mask,
        )
        out = self.transformer.decoder(
            tgt,
            memory,
            tgt_mask=tgt_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        logits = self.generator(out)  # (B, T, vocab)
        return logits

    @torch.no_grad()
    def greedy_decode(self, src_tokens: torch.Tensor, src_key_padding_mask: Optional[torch.Tensor],
                      max_len: int, bos_id: int, eos_id: int) -> torch.Tensor:
        B = src_tokens.size(0)
        ys = torch.full((B, 1), bos_id, dtype=torch.long, device=src_tokens.device)
        for _ in range(max_len - 1):
            logits = self.forward(src_tokens, src_key_padding_mask, ys, None)
            next_token = logits[:, -1, :].argmax(-1, keepdim=True)
            ys = torch.cat([ys, next_token], dim=1)
            if (next_token == eos_id).all():
                break
        return ys
