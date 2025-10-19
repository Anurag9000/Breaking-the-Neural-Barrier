import torch
import torch.nn as nn
import torch.nn.functional as F
import random

from model_t5_span_infilling import Encoder as Enc, Decoder as Dec

class BARTDenoise(nn.Module):
    """Seq2seq denoising: token deletion, masking, and sentence permutation."""
    def __init__(self, vocab, dim=512, enc_depth=6, dec_depth=6, heads=8, mlp_ratio=4.0, max_len=256):
        super().__init__()
        self.enc = Enc(vocab, dim, enc_depth, heads, mlp_ratio, max_len)
        self.dec = Dec(vocab, dim, dec_depth, heads, mlp_ratio, max_len)

    def corrupt(self, x):
        B,L = x.shape
        out=[]
        for b in range(B):
            toks = x[b].tolist()
            # deletion
            toks = [t for t in toks if random.random() > 0.1 or t==0]
            # mask some
            toks = [103 if (random.random()<0.1 and t!=0) else t for t in toks]
            out.append(torch.tensor(toks[:L], device=x.device))
        max_len = max(t.size(0) for t in out)
        X = torch.zeros(B, max_len, dtype=torch.long, device=x.device)
        for b in range(B): X[b,:out[b].size(0)] = out[b]
        return X

    def forward(self, x):
        src = self.corrupt(x)
        mem = self.enc(src)
        logits = self.dec(x[:, :-1], mem)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), x[:, 1:].reshape(-1))
        return loss
