# models/bernoulli_diffusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.f_token_unet import TokenUNet


class BernoulliDiffusion(nn.Module):
    """
    Diffusion model for binary images (or per-channel binary token embeddings).
    Uses a TokenUNet with C=1, K=2 embeddings.
    """
    def __init__(self, T=1000, beta_min=1e-4, beta_max=0.02, base=64):
        super().__init__()
        self.T = T
        self.beta_min = beta_min
        self.beta_max = beta_max

        # UNet for 1-channel, 2-token embeddings
        self.net = TokenUNet(embed_dim=32, base=base, channels=1, K=2)
        self.emb = nn.Embedding(2, 32)

    def beta(self, t):
        """
        Linear schedule for flipping probability at timestep t.
        t: tensor [B]
        """
        return self.beta_min + (self.beta_max - self.beta_min) * (t.float() / self.T)

    def binarize(self, x):
        """
        Convert [-1,1] image to binary {0,1} by channel mean thresholding.
        x: [B,C,H,W]
        """
        return (x.mean(dim=1, keepdim=True) > 0).long()

    def q_sample(self, x0, t):
        """
        Corrupt binary image x0 at timestep t by flipping bits with probability beta_t.
        """
        beta_t = self.beta(t).view(-1, 1, 1, 1)
        flip = (torch.rand_like(x0.float()) < beta_t).long()
        return x0 ^ flip

    def loss(self, x_img):
        """
        Compute cross-entropy loss between predicted logits and true binary tokens.
        x_img: [-1,1] input image [B,C,H,W]
        """
        x0 = self.binarize(x_img)  # [B,1,H,W]
        B, _, H, W = x0.shape
        device = x0.device

        t = torch.randint(0, self.T, (B,), device=device)
        x_t = self.q_sample(x0, t)

        # Embed tokens
        x_emb = self.emb(x_t.squeeze(1)).permute(0, 3, 1, 2)  # [B,32,H,W]
        logits = self.net(x_emb, t)  # [B,2,H,W] (C*K with C=1,K=2)
        logits = logits.view(B, 2, H, W)

        # Flatten for cross-entropy
        logits_flat = logits.permute(0, 2, 3, 1).reshape(-1, 2)
        target_flat = x0.view(-1)
        return F.cross_entropy(logits_flat, target_flat)

    @torch.no_grad()
    def sample(self, B, steps=100, device='cuda', size=(1, 32, 32)):
        """
        Sample binary images using the Bernoulli diffusion process.
        Returns images expanded to 3 channels in [-1,1] for visualization.
        """
        _, H, W = size
        x = torch.randint(0, 2, (B, 1, H, W), device=device)

        for s in reversed(range(steps)):
            t = torch.full((B,), int(self.T * s / steps), device=device, dtype=torch.long)
            x_emb = self.emb(x.squeeze(1)).permute(0, 3, 1, 2)
            logits = self.net(x_emb, t).view(B, 2, H, W)

            # Greedy x0 prediction
            x0_hat = logits.permute(0, 2, 3, 1).argmax(dim=-1, keepdim=True)

            # Re-corrupt toward previous step
            beta_t = self.beta(t).view(-1, 1, 1, 1)
            mask = (torch.rand_like(x.float()) < beta_t).long()
            x = x0_hat * mask + x * (1 - mask)

        # Expand to 3-channel image in [-1,1]
        return (x.float() * 2 - 1).repeat(1, 3, 1, 1)
