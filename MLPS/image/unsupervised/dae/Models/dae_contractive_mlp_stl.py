import torch
import torch.nn as nn
from typing import List, Tuple

from dae_saltpepper_mlp_stl import dae_total_neurons


class DAEContractiveMLP(nn.Module):
    """
    Fully-connected denoising autoencoder with a contractive-style penalty
    on encoder weights. Architecture is a standard MLP encoder/decoder.
    """

    def __init__(
        self,
        in_channels: int = 3,
        img_size: int = 32,
        width: int = 512,
        depth: int = 3,
    ):
        super().__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.img_size = img_size
        self.width = width
        self.depth = depth

        input_dim = in_channels * img_size * img_size
        self.input_dim = input_dim

        enc_layers: List[nn.Module] = []
        enc_layers.append(nn.Linear(input_dim, width))
        enc_layers.append(nn.ReLU(inplace=True))
        for _ in range(depth - 1):
            enc_layers.append(nn.Linear(width, width))
            enc_layers.append(nn.ReLU(inplace=True))
        self.encoder = nn.Sequential(*enc_layers)

        dec_layers: List[nn.Module] = []
        for _ in range(depth - 1):
            dec_layers.append(nn.Linear(width, width))
            dec_layers.append(nn.ReLU(inplace=True))
        dec_layers.append(nn.Linear(width, input_dim))
        self.decoder = nn.Sequential(*dec_layers)

        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def contractive_loss(self, x: torch.Tensor) -> torch.Tensor:
        """
        Computes the layer-wise contractive penalty: || J(x) ||_F^2
        For ReLU: sum_j [ I(z_j > 0) * ||W_j||^2 ]
        """
        b = x.size(0)
        h = x.view(b, self.input_dim)
        total_loss = torch.tensor(0.0, device=x.device)
        
        # Iterate through encoder layers (Linear -> ReLU) pair
        # We assume structure: Linear -> ReLU -> Linear -> ReLU ...
        # We can iterate children.
        
        # Since nn.Sequential logic is hidden, we manually iterate
        # to get pre-activations.
        
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                # Store weight for next activation check
                # shape: (out, in)
                # row norm squared: ||W_j||^2
                w_sq = layer.weight.pow(2).sum(dim=1) # (out,)
                h = layer(h) # Pre-activation z
            elif isinstance(layer, nn.ReLU):
                # h is pre-activation z
                # Mask: z > 0 (shape b, out)
                mask = (h > 0).float()
                # Penalty for this layer:
                # sum over batch, sum over output units j: mask_bj * w_sq_j
                # Average over batch? Usually sum or mean.
                # Paper implies expectation over data distribution.
                # We'll use sum over batch (standard), or mean.
                # Let's match reconstruction loss reduction (usually mean or sum).
                # Current code used sum of weights (scalar).
                # New code: sum_b ( batch_loss ). mean over batch?
                # Let's use mean over batch to be scale-invariant.
                
                layer_loss = (mask * w_sq.unsqueeze(0)).sum(dim=1).mean()
                total_loss += layer_loss
                h = layer(h) # Post-activation
            else:
                h = layer(h)
                
        return total_loss

    def encoder_linears(self) -> List[nn.Linear]:
        return [m for m in self.encoder if isinstance(m, nn.Linear)]

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        b = x.size(0)
        z_in = x.view(b, self.input_dim)
        return self.encoder(z_in)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        b = z.size(0)
        out = self.decoder(z).view(b, self.in_channels, self.img_size, self.img_size)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        z = self.encode(x)
        x_rec = self.decode(z)
        return x_rec, z


__all__ = ["DAEContractiveMLP", "dae_total_neurons"]
