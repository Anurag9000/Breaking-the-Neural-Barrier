
import torch.nn as nn
try:
    from .nlp_ae_common import MLPBlock, TextAvgEmbed
except ImportError:
    from nlp_ae_common import MLPBlock, TextAvgEmbed

class MLPTextAE(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: list, rep_dim: int, use_bn: bool=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden = list(hidden)
        self.rep_dim = rep_dim
        self.use_bn = use_bn

        self.encoder_tok = TextAvgEmbed(vocab_size, emb_dim)
        layers = []
        prev = emb_dim
        for w in hidden:
            layers.append(MLPBlock(prev, w, use_bn))
            prev = w
        self.backbone = nn.Sequential(*layers)
        self.rep = nn.Linear(prev, rep_dim)
        self.decoder = nn.Linear(rep_dim, vocab_size)  # logits over vocab

    def forward(self, view):
        token_ids, lengths = view
        h0 = self.encoder_tok(token_ids, lengths)
        h = self.backbone(h0) if len(self.backbone) > 0 else h0
        z = self.rep(h)
        logits = self.decoder(z)
        return logits  # use with soft CE to BOW TF targets
