import torch
import torch.nn as nn

class XTransformer(nn.Module):
    """Label-attentive classifier for extreme multi-label classification (single-model).
    Encodes document tokens with Transformer encoder and attends over learned label embeddings.
    """
    def __init__(self, vocab, num_labels, d_model=256, nhead=8, depth=6):
        super().__init__()
        self.emb = nn.Embedding(vocab, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.label_emb = nn.Embedding(num_labels, d_model)
        self.score = nn.Linear(d_model, 1)
    def forward(self, ids):
        B,S = ids.shape
        x = self.encoder(self.emb(ids))  # B,S,D
        doc = x.mean(dim=1, keepdim=True)  # B,1,D
        # attend label embeddings with doc as query
        labels = self.label_emb.weight.unsqueeze(0).expand(B, -1, -1)  # B,L,D
        attn = (doc @ labels.transpose(1,2)) / (labels.size(-1)**0.5)  # B,1,L
        w = attn.softmax(-1)
        z = (w @ labels).squeeze(1)  # B,D
        # final score per label via dot with each label embedding
        logits = z @ self.label_emb.weight.t()  # B,L
        return logits
