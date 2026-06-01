
import torch
import torch.nn as nn
try:
    from .nlp_ssl_common import MLPBlock, TextAvgEmbed, nt_xent_loss
except ImportError:
    from nlp_ssl_common import MLPBlock, TextAvgEmbed, nt_xent_loss

class MLPTextSSL(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: list, rep_dim: int, proj_dim: int, use_bn: bool=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden = list(hidden)
        self.rep_dim = rep_dim
        self.proj_dim = proj_dim
        self.use_bn = use_bn

        self.encoder_tok = TextAvgEmbed(vocab_size, emb_dim)
        blocks = []
        prev = emb_dim
        for w in hidden:
            blocks.append(MLPBlock(prev, w, use_bn)); prev = w
        self.backbone = nn.Sequential(*blocks)
        self.rep = nn.Linear(prev, rep_dim)
        self.proj = nn.Sequential(
            nn.Linear(rep_dim, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )

    def encode(self, view):
        tok, lens = view
        h0 = self.encoder_tok(tok, lens)
        h = self.backbone(h0) if len(self.backbone) > 0 else h0
        z = self.rep(h)
        p = self.proj(z)
        return p

    def forward(self, v1, v2, temperature=0.05):
        z1 = self.encode(v1); z2 = self.encode(v2)
        return nt_xent_loss(z1, z2, temperature=temperature)
