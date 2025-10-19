import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from model_t5_span_infilling import Encoder as Enc, Decoder as Dec

class MASSSeq2Seq(nn.Module):
    """MASS: Masked Sequence to Sequence pretraining.
    Mask a continuous span in input; decoder predicts that span autoregressively.
    """
    def __init__(self, vocab, dim=512, enc_depth=6, dec_depth=6, heads=8, mlp_ratio=4.0, max_len=256):
        super().__init__()
        self.enc = Enc(vocab, dim, enc_depth, heads, mlp_ratio, max_len)
        self.dec = Dec(vocab, dim, dec_depth, heads, mlp_ratio, max_len)
        self.mask_id = 103

    def make_mass(self, x, span_ratio=0.5):
        B,L = x.shape
        X = x.clone(); spans=[]
        for b in range(B):
            span = max(1, int(L*span_ratio*random.uniform(0.3, 0.7)))
            s = random.randint(0, max(0, L-span))
            X[b, s:s+span] = self.mask_id
            spans.append(x[b, s:s+span])
        # build decoder inputs/targets from spans
        max_s = max(t.size(0) for t in spans)
        Y = torch.zeros(B, max_s, dtype=torch.long, device=x.device)
        for b in range(B): Y[b, :spans[b].size(0)] = spans[b]
        return X, Y

    def forward(self, x):
        src, tgt = self.make_mass(x)
        mem = self.enc(src)
        logits = self.dec(tgt[:, :-1], mem)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        return loss
