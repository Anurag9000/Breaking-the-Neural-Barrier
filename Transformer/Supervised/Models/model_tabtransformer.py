import torch
import torch.nn as nn

class TabTransformer(nn.Module):
    """TabTransformer for categorical tabular classification.
    - Embeds each categorical feature as a token.
    - Applies Transformer encoder over tokens.
    - Pooled representation -> classifier.
    """
    def __init__(self, num_categories_per_field, num_classes, d_model=128, nhead=8, depth=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(nc, d_model) for nc in num_categories_per_field])
        enc = nn.TransformerEncoderLayer(d_model, nhead, int(d_model*mlp_ratio), dropout, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)

    def forward(self, x_cat):
        # x_cat: (B, F) integer indices
        tokens = [emb(x_cat[:, i]) for i, emb in enumerate(self.embs)]
        z = torch.stack(tokens, dim=1)  # (B, F, D)
        z = self.encoder(z)
        z = self.norm(z.mean(dim=1))
        return self.head(z)
