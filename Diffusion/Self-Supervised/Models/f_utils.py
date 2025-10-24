# models/f_utils.py
import torch
import torch.nn as nn
import torch.nn.functional as F


# -----------------------
# Pixel Tokenizer
# -----------------------
class Tokenizer(nn.Module):
    """
    Quantize pixel values in [-1,1] into K bins per channel (uniform).
    """
    def __init__(self, K=16):
        super().__init__()
        self.K = K

    def encode(self, x):
        """
        x: tensor in [-1,1], shape [B,C,H,W]
        Returns: integer indices in {0,..,K-1}, same shape
        """
        # Map [-1,1] -> [0,1]
        x01 = (x.clamp(-1, 1) + 1) / 2
        # Quantize to K bins
        idx = torch.clamp((x01 * self.K).floor().long(), 0, self.K - 1)
        return idx

    def decode(self, idx):
        """
        idx: integer indices in {0,..,K-1}, shape [B,C,H,W]
        Returns: reconstructed tensor in [-1,1], same shape
        """
        x01 = (idx.float() + 0.5) / self.K
        return x01 * 2 - 1


# -----------------------
# Token Embedding
# -----------------------
class TokenEmbedding(nn.Module):
    """
    Embed tokenized pixel indices into vectors.
    """
    def __init__(self, K=16, embed_dim=64, channels=3):
        super().__init__()
        self.K = K
        self.channels = channels
        self.emb = nn.Embedding(K * channels, embed_dim)

    def forward(self, idx):
        """
        idx: [B,C,H,W] in {0,..,K-1}
        Returns: [B, embed_dim, C, H, W] tensor of embeddings
        """
        B, C, H, W = idx.shape
        # Offset each channel to have distinct vocabulary
        offsets = torch.arange(C, device=idx.device).view(1, C, 1, 1) * self.K
        tok = (idx + offsets).view(B, -1)
        emb = self.emb(tok)
        # Reshape to [B, embed_dim, C, H, W]
        emb = emb.view(B, C, H, W, -1).permute(0, 4, 1, 2, 3).contiguous()
        # If embed_dim is 1, squeeze the last dimension
        return emb.squeeze(-1) if emb.dim() == 5 else emb


# -----------------------
# Logits reshaping
# -----------------------
def logits_to_per_channel(logits, C, K):
    """
    Reshape logits from [B, C*K, H, W] -> [B, C, K, H, W]
    """
    B, CK, H, W = logits.shape
    return logits.view(B, C, K, H, W)
