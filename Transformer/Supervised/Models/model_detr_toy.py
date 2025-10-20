import math
from typing import Optional
import torch
import torch.nn as nn

class TinyBackbone(nn.Module):
    def __init__(self, in_ch=3, dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 64, 7, 2, 3), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1), nn.BatchNorm2d(128), nn.ReLU(inplace=True),
            nn.Conv2d(128, dim, 3, 1, 1), nn.BatchNorm2d(dim), nn.ReLU(inplace=True),
        )
    def forward(self, x):
        return self.net(x)  # (B, C, H/4, W/4)

class PositionalEncoding2D(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.row = nn.Parameter(torch.randn(1, d_model//2))
        self.col = nn.Parameter(torch.randn(1, d_model//2))
    def forward(self, H, W):
        r = self.row.repeat(H, 1)
        c = self.col.repeat(W, 1)
        pe = torch.cat([
            r.unsqueeze(1).repeat(1, W, 1),
            c.unsqueeze(0).repeat(H, 1, 1),
        ], dim=-1)  # H,W,D
        return pe

class DETRToy(nn.Module):
    """Single-model DETR-style detector (toy): tiny CNN backbone + Transformer encoder/decoder + object queries."""
    def __init__(self, num_classes: int, hidden_dim: int = 128, nheads: int = 4, enc_layers: int = 3, dec_layers: int = 3, num_queries: int = 50):
        super().__init__()
        self.backbone = TinyBackbone(dim=hidden_dim)
        self.input_proj = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=1)
        self.pos2d = PositionalEncoding2D(hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(hidden_dim, nheads, hidden_dim*4, 0.1, batch_first=True, norm_first=True)
        dec_layer = nn.TransformerDecoderLayer(hidden_dim, nheads, hidden_dim*4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc_layer, enc_layers)
        self.decoder = nn.TransformerDecoder(dec_layer, dec_layers)
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.class_head = nn.Linear(hidden_dim, num_classes)
        self.bbox_head = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(), nn.Linear(hidden_dim, 4), nn.Sigmoid())

    def forward(self, x):
        B, C, H, W = x.shape
        feat = self.backbone(x)
        feat = self.input_proj(feat)
        _, _, Hf, Wf = feat.shape
        pe = self.pos2d(Hf, Wf).to(feat.device).view(1, Hf*Wf, -1)
        src = feat.flatten(2).transpose(1,2) + pe  # (B, HW, D)
        memory = self.encoder(src)
        queries = self.query_embed.weight.unsqueeze(0).expand(B, -1, -1)
        hs = self.decoder(queries, memory)
        logits = self.class_head(hs)  # (B, Q, C)
        boxes = self.bbox_head(hs)    # (B, Q, 4) normalized cx,cy,w,h
        return logits, boxes
