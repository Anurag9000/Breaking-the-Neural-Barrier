import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvFeature(nn.Module):
    def __init__(self, in_ch=1, channels=(32,64,128)):
        super().__init__()
        layers=[]; c=in_ch
        for ch in channels:
            layers += [nn.Conv2d(c, ch, 3, 1, 1), nn.BatchNorm2d(ch), nn.ReLU(), nn.MaxPool2d((2,2))]
            c=ch
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x)

class TransformerEncoder1D(nn.Module):
    def __init__(self, d_model=256, nhead=8, depth=4):
        super().__init__()
        self.depth = depth
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
    def forward(self, x):
        return self.encoder(x)

class ConvTransformerCTC(nn.Module):
    """Single-model ASR toy: spectrogram -> conv -> sequence -> Transformer -> CTC logits."""
    def __init__(self, vocab=30, d_model=256, depth=4):
        super().__init__()
        self.vocab = vocab
        self.d_model = d_model
        self.depth = depth
        self.conv = ConvFeature(1, (32,64,128))
        self.proj = nn.Linear(128*4, d_model)
        self.tr = TransformerEncoder1D(d_model, 8, depth)
        self.head = nn.Linear(d_model, vocab)
    def forward(self, spec):
        # spec: (B,1,T,F) -> conv -> (B,C,T/8,F/8)
        x = self.conv(spec)
        B,C,T,Fq = x.shape
        x = x.permute(0,2,1,3).contiguous().view(B,T, C*Fq//4)  # reduce feat dim modestly
        x = self.proj(x)
        x = self.tr(x)
        return self.head(x)  # (B, T, vocab)
