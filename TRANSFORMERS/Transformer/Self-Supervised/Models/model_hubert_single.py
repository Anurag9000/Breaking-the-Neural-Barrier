import torch
import torch.nn as nn
import torch.nn.functional as F

from model_wav_2_vec_2_single import ConvFeatureEncoder

class HuBERTSingle(nn.Module):
    """HuBERT-style with offline k-means targets over MFCC-like features.
    We approximate MFCCs by a small conv front-end and freeze k-means centroids built at start.
    Single model (no teacher)."""
    def __init__(self, code_k=500, feat_dim=64):
        super().__init__()
        self.feature = ConvFeatureEncoder(in_ch=1, dims=(64,64), strides=(5,2), kernels=(10,3))
        self.proj = nn.Linear(64, feat_dim)
        self.context = nn.TransformerEncoder(nn.TransformerEncoderLayer(feat_dim, 8, 512, batch_first=True), 6)
        self.codebook = nn.Parameter(torch.randn(code_k, feat_dim), requires_grad=False)
        self.pred = nn.Linear(feat_dim, code_k)
        self.mask_prob = 0.065; self.mask_len=10

    @torch.no_grad()
    def init_codebook(self, feats, iters=10):
        C = self.codebook.size(0)
        idx = torch.randperm(feats.size(0), device=feats.device)[:C]
        self.codebook.copy_(feats[idx])
        for _ in range(iters):
            d = torch.cdist(feats, self.codebook)
            a = d.argmin(dim=1)
            for c in range(C):
                m = (a==c)
                if m.any():
                    self.codebook[c] = feats[m].mean(dim=0)

    def mask(self, T, device):
        m = torch.zeros(T, dtype=torch.bool, device=device)
        num = max(1, int(self.mask_prob * T))
        for _ in range(num):
            s = torch.randint(0, T, (1,), device=device)
            e = (s + self.mask_len).clamp(max=T)
            m[s:e] = True
        return m

    def forward(self, wav):
        z = self.feature(wav)          # (B,T',C)
        z = self.proj(z)               # (B,T',D)
        B,T,D = z.shape
        ctx = self.context(z)
        loss=0.0; n=0
        for b in range(B):
            m = self.mask(T, wav.device)
            if not m.any():
                continue
            # assign offline targets by nearest centroid of z (stop-grad)
            with torch.no_grad():
                d = torch.cdist(z[b][m], self.codebook)
                labels = d.argmin(dim=1)
            logits = self.pred(ctx[b][m])
            loss += F.cross_entropy(logits, labels); n+=1
        return loss/max(n,1)
