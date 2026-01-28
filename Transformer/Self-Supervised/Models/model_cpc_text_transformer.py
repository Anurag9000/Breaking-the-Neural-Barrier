import torch
import torch.nn as nn
import torch.nn.functional as F
from model_simcse_transformer import TransformerEncoder, SimpleTokenizer

class CPCTxtTransformer(nn.Module):
    """Contrastive Predictive Coding on text latents with a Transformer context model.
    Single encoder used for both context and future latents; prediction with linear heads.
    """
    def __init__(self, vocab, dim=256, depth=4, heads=8, mlp_ratio=4.0, steps_ahead=3):
        super().__init__()
        self.vocab = vocab
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.mlp_ratio = mlp_ratio
        
        self.enc = TransformerEncoder(vocab, dim, depth, heads, mlp_ratio)
        self.predictors = nn.ModuleList([nn.Linear(dim, dim) for _ in range(steps_ahead)])
        self.steps_ahead = steps_ahead

    def forward(self, x):
        # x: (B,L)
        h = self.enc.emb(x)
        h = self.enc.pos(h)
        for lyr in self.enc.layers:
            h = lyr(h)  # (B,L,D)
        h = self.enc.norm(h)
        B,L,D = h.shape
        loss = 0.0
        for k in range(1, self.steps_ahead+1):
            pred = self.predictors[k-1](h[:, :-k])  # (B, L-k, D)
            target = h[:, k:]
            pred = F.normalize(pred, dim=-1); target = F.normalize(target, dim=-1)
            logits = pred @ target.transpose(1,2)  # (B, L-k, L-k)
            labels = torch.arange(logits.size(1), device=x.device).unsqueeze(0).repeat(B,1)
            loss += F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
        return loss / self.steps_ahead
