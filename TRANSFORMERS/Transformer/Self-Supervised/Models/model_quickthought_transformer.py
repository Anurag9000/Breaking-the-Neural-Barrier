import torch
import torch.nn as nn
import torch.nn.functional as F
from model_simcse_transformer import TransformerEncoder, SimpleTokenizer

class QuickThoughtTransformer(nn.Module):
    """QuickThought-like objective: given a sentence, pick its true neighbor among negatives.
    Single encoder; we build (anchor, positive, negatives) batches.
    """
    def __init__(self, vocab, dim=256, depth=4, heads=8, mlp_ratio=4.0):
        super().__init__()
        self.encoder = TransformerEncoder(vocab, dim, depth, heads, mlp_ratio)

    def forward(self, anchors, candidates):
        # anchors: (B,L), candidates: (B,K,L) with K options (first is positive)
        B,K,L = candidates.shape
        a = self.encoder(anchors)  # (B,D)
        cand = self.encoder(candidates.view(B*K, L)).view(B, K, -1)  # (B,K,D)
        a = F.normalize(a, dim=-1)
        cand = F.normalize(cand, dim=-1)
        logits = (cand @ a.unsqueeze(-1)).squeeze(-1)  # (B,K)
        targets = torch.zeros(B, dtype=torch.long, device=anchors.device)  # index 0 is positive
        loss = F.cross_entropy(logits, targets)
        return loss
