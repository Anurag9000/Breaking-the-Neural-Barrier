import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class PosEnc(nn.Module):
    def __init__(self, dim, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, dim)
        pos = torch.arange(0, max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, dim, 2)*(-torch.log(torch.tensor(10000.0))/dim))
        pe[:, 0::2] = torch.sin(pos*div); pe[:, 1::2] = torch.cos(pos*div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Encoder(nn.Module):
    def __init__(self, vocab, dim, depth, heads, mlp_ratio, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = PosEnc(dim, max_len)
        self.layers = nn.ModuleList([nn.TransformerEncoderLayer(dim, heads, int(dim*mlp_ratio), batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
    def forward(self, x):
        h = self.pos(self.emb(x))
        for lyr in self.layers: h = lyr(h)
        return self.norm(h)

class Decoder(nn.Module):
    def __init__(self, vocab, dim, depth, heads, mlp_ratio, max_len):
        super().__init__()
        self.emb = nn.Embedding(vocab, dim)
        self.pos = PosEnc(dim, max_len)
        self.layers = nn.ModuleList([nn.TransformerDecoderLayer(dim, heads, int(dim*mlp_ratio), batch_first=True) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab)
    def forward(self, y, mem):
        B,L = y.shape
        tgt_mask = torch.triu(torch.ones(L,L, device=y.device), diagonal=1).bool()
        h = self.pos(self.emb(y))
        for lyr in self.layers:
            h = lyr(h, mem, tgt_mask=tgt_mask)
        h = self.norm(h)
        return self.head(h)

class T5SpanInfilling(nn.Module):
    """T5-style span corruption with sentinel tokens. Single-model seq2seq."""
    def __init__(self, vocab, dim=512, enc_depth=6, dec_depth=6, heads=8, mlp_ratio=4.0, max_len=256, sent_start=32000):
        super().__init__()
        self.enc = Encoder(vocab, dim, enc_depth, heads, mlp_ratio, max_len)
        self.dec = Decoder(vocab, dim, dec_depth, heads, mlp_ratio, max_len)
        self.sent_start = sent_start

    def make_spans(self, x, span_prob=0.15, mean_span=3):
        # build input with sentinels and target as concatenation of masked spans prefixed by sentinel
        B,L = x.shape
        inputs=[]; targets=[]
        for b in range(B):
            toks = x[b].tolist()
            i=0; masked=[]; inp=[]; tgt=[]; sid=self.sent_start
            while i < L:
                if random.random() < span_prob:
                    span = max(1, int(random.expovariate(1/mean_span)))
                    inp.append(sid); masked.extend(toks[i:i+span])
                    tgt.append(sid); tgt.extend(toks[i:i+span])
                    sid += 1; i += span
                else:
                    inp.append(toks[i]); i+=1
            inputs.append(torch.tensor(inp[:L], device=x.device))
            targets.append(torch.tensor(tgt[:L], device=x.device))
        # pad to equal length
        max_in = max(t.size(0) for t in inputs); max_tg = max(t.size(0) for t in targets)
        X = torch.zeros(B, max_in, dtype=torch.long, device=x.device)
        Y = torch.zeros(B, max_tg, dtype=torch.long, device=x.device)
        for b in range(B):
            X[b,:inputs[b].size(0)] = inputs[b]
            Y[b,:targets[b].size(0)] = targets[b]
        return X, Y

    def forward(self, x):
        src, tgt = self.make_spans(x)
        mem = self.enc(src)
        logits = self.dec(tgt[:, :-1], mem)
        loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt[:, 1:].reshape(-1))
        return loss
