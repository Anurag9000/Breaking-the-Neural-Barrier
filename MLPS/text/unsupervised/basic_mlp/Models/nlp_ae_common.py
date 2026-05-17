
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
        if self.bn is not None: x = self.bn(x)
        return self.act(x)

class TextAvgEmbed(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
    def forward(self, token_ids, lengths):
        E = self.emb(token_ids)                              # (B,L,D)
        mask = (token_ids != 0).float().unsqueeze(-1)        # (B,L,1)
        summed = (E * mask).sum(dim=1)                       # (B,D)
        denom = lengths.clamp(min=1).float().unsqueeze(-1)   # (B,1)
        return summed / denom                                # (B,D)

def soft_ce_loss(logits, targets, reduction="mean"):
    """
    Cross-entropy against soft targets (each row sums to 1).
    logits: (B,V) raw scores; targets: (B,V) soft TF distribution.
    """
    logp = F.log_softmax(logits, dim=-1)
    loss = -(targets * logp).sum(dim=-1)  # (B,)
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss
