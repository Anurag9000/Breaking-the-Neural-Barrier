# models/multinomial_masked.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.f_token_unet import TokenUNet
from models.f_utils import Tokenizer, TokenEmbedding, logits_to_per_channel


class MultinomialMasked(nn.Module):
    """
    Multinomial masked diffusion model.
    - Uses token masking with a special MASK token.
    - Predicts original tokens using a TokenUNet over embeddings.
    """
    def __init__(self, K=16, channels=3, T=1000, beta_min=1e-4, beta_max=0.02,
                 embed_dim=64, base=64):
        super().__init__()
        self.K = K
        self.C = channels
        self.T = T
        self.mask_id = K  # extra MASK token
        self.beta_min = beta_min
        self.beta_max = beta_max

        # Tokenizer for original pixels
        self.tokenizer = Tokenizer(K)
        # Embedding includes MASK token
        self.embed = TokenEmbedding(K + 1, embed_dim, channels)
        self.net = TokenUNet(embed_dim=embed_dim, base=base, channels=channels, K=K + 1)

    def beta(self, t):
        """Linear schedule for masking probability."""
        return self.beta_min + (self.beta_max - self.beta_min) * (t.float() / self.T)

    def q_sample(self, x0_idx, t):
        """
        Corrupt tokens at timestep t by replacing with MASK token with probability beta_t.
        """
        B, C, H, W = x0_idx.shape
        beta_t = self.beta(t).view(-1, 1, 1, 1)
        mask = (torch.rand_like(x0_idx.float()) < beta_t).long()
        x_t = x0_idx * (1 - mask) + self.mask_id * mask
        return x_t

    def loss(self, x_img):
        """
        Compute cross-entropy loss between predicted logits and original token indices.
        Ignores MASK token in target.
        """
        x0_idx = self.tokenizer.encode(x_img)  # [B,C,H,W]
        B, device = x0_idx.size(0), x0_idx.device
        t = torch.randint(0, self.T, (B,), device=device)
        x_t = self.q_sample(x0_idx, t)
        x_emb = self.embed(x_t)
        logits = self.net(x_emb, t)  # [B, C*(K+1), H, W]
        logits = logits_to_per_channel(logits, self.C, self.K + 1)

        # Flatten for cross-entropy
        logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, self.K + 1)
        target_flat = x0_idx.permute(0, 1, 3, 4).reshape(-1)
        return F.cross_entropy(logits_flat, target_flat)

    @torch.no_grad()
    def sample(self, B, steps=100, device='cuda', size=(3, 32, 32)):
        """
        Sample images using masked diffusion.
        Starts fully masked and gradually replaces tokens according to beta_t schedule.
        Returns images in [-1,1].
        """
        C, H, W = size
        x_idx = torch.full((B, C, H, W), self.mask_id, device=device, dtype=torch.long)

        for s in reversed(range(steps)):
            t = torch.full((B,), int(self.T * s / steps), device=device, dtype=torch.long)
            x_emb = self.embed(x_idx)
            logits = self.net(x_emb, t)
            logits = logits_to_per_channel(logits, C, self.K + 1)

            # Predict original tokens, ignoring MASK token
            probs = torch.softmax(logits, dim=2)
            x0_hat = probs[..., :self.K].argmax(dim=2)

            # Unmask a fraction of tokens according to beta_t
            beta_t = self.beta(t).view(-1, 1, 1, 1)
            unmask = (torch.rand_like(x_idx.float()) < beta_t).long()
            x_idx = x_idx * (1 - unmask) + x0_hat * unmask

        return ((x_idx.float().clamp(max=self.K - 1) + 0.5) / self.K * 2 - 1).clamp(-1, 1)
