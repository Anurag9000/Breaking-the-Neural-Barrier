# models/d3pm_masked.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.f_token_unet import TokenUNet
from models.f_utils import Tokenizer, TokenEmbedding, logits_to_per_channel


class D3PMMasked(nn.Module):
    """
    D3PM with masked/uniform token corruption.
    Works on tokenized pixel embeddings.
    """
    def __init__(self, K=16, channels=3, T=1000, beta_min=1e-4, beta_max=0.02,
                 embed_dim=64, base=64):
        super().__init__()
        self.K = K
        self.C = channels
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max

        self.tokenizer = Tokenizer(K)
        self.embed = TokenEmbedding(K, embed_dim, channels)
        self.net = TokenUNet(embed_dim=embed_dim, base=base, channels=channels, K=K)

    def beta(self, t):
        """
        Linear schedule for masking probability at timestep t.
        t: tensor of shape [B]
        Returns: tensor [B] of beta values
        """
        return self.beta_min + (self.beta_max - self.beta_min) * (t.float() / self.T)

    def q_sample(self, x0_idx, t):
        """
        Masked/uniform replacement corruption for timestep t.
        x0_idx: token indices [B,C,H,W]
        t: tensor [B] of timesteps
        Returns: corrupted tokens x_t [B,C,H,W]
        """
        B, C, H, W = x0_idx.shape
        beta_t = self.beta(t).view(-1, 1, 1, 1)
        mask = (torch.rand_like(x0_idx.float()) < beta_t).long()
        random_tok = torch.randint(0, self.K, x0_idx.shape, device=x0_idx.device)
        x_t = x0_idx * (1 - mask) + random_tok * mask
        return x_t

    def loss(self, x_img):
        """
        Compute cross-entropy loss between predicted logits and true tokens.
        x_img: input image tensor [-1,1], shape [B,C,H,W]
        """
        x0_idx = self.tokenizer.encode(x_img)
        B = x0_idx.size(0)
        device = x0_idx.device

        t = torch.randint(0, self.T, (B,), device=device)
        x_t = self.q_sample(x0_idx, t)
        x_emb = self.embed(x_t)
        logits = self.net(x_emb, t)  # [B, C*K, H, W]
        logits = logits_to_per_channel(logits, self.C, self.K)

        # Flatten for cross-entropy
        logits_flat = logits.permute(0, 1, 3, 4, 2).reshape(-1, self.K)
        target_flat = x0_idx.permute(0, 1, 3, 4).reshape(-1)
        return F.cross_entropy(logits_flat, target_flat)

    @torch.no_grad()
    def sample(self, B, steps=100, device='cuda', size=(3, 32, 32)):
        """
        Sample images using the D3PMMasked model.
        Returns: tensor [B, C, H, W] in [-1,1]
        """
        C, H, W = size
        x_idx = torch.randint(0, self.K, (B, C, H, W), device=device)

        for s in reversed(range(steps)):
            t = torch.full((B,), int(self.T * s / steps), device=device, dtype=torch.long)
            x_emb = self.embed(x_idx)
            logits = self.net(x_emb, t)
            logits = logits_to_per_channel(logits, C, self.K)

            # Greedy x0 prediction
            x0_hat = logits.permute(0, 1, 3, 4, 2).argmax(dim=-1)

            # Re-corrupt toward x_{t-1} with beta_t
            beta_t = self.beta(t).view(-1, 1, 1, 1)
            mask = (torch.rand_like(x_idx.float()) < beta_t).long()
            x_idx = x0_hat * mask + x_idx * (1 - mask)

        # Decode tokens to [-1,1]
        return ((x_idx.float() + 0.5) / self.K * 2 - 1).clamp(-1, 1)
