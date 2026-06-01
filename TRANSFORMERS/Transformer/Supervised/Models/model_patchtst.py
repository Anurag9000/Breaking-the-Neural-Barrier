import torch
import torch.nn as nn

class PatchTST(nn.Module):
    """PatchTST: channel-independent patching + transformer for multivariate TS classification/forecasting.
    Here: classification head (average pooled CLS over patches).
    """
    def __init__(self, num_series: int, num_classes: int, patch_len: int = 16, stride: int = 8, d_model: int = 256, nhead: int = 8, depth: int = 6, mlp_ratio: float = 4.0):
        super().__init__()
        self.patch_len = patch_len
        self.stride = stride
        self.token = nn.Linear(patch_len, d_model)
        enc_layer = nn.TransformerEncoderLayer(d_model, nhead, int(d_model*mlp_ratio), 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, depth)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, num_classes)
        self.num_series = num_series

    def forward(self, x):
        # x: (B, T, C) with C=num_series
        B, T, C = x.shape
        # unfold along time per channel
        patches = []
        for c in range(C):
            seq = x[:, :, c]  # (B, T)
            # (B, N, patch_len)
            N = 1 + (T - self.patch_len) // self.stride
            windows = seq.unfold(dimension=1, size=self.patch_len, step=self.stride)
            patches.append(windows)
        patches = torch.stack(patches, dim=2)  # (B, N, C, P)
        B, N, C, P = patches.shape
        tokens = self.token(patches.view(B*N*C, P)).view(B, N*C, -1)
        z = self.encoder(tokens)
        z = self.norm(z.mean(dim=1))
        return self.head(z)
