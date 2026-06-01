import torch
import torch.nn as nn

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super().__init__(); self.chomp_size = chomp_size
    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous() if self.chomp_size>0 else x

class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size, stride=stride, padding=padding, dilation=dilation)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = self.chomp1(out)
        out = self.relu1(out)
        out = self.drop1(out)

        out = self.conv2(out)
        out = self.chomp2(out)
        out = self.relu2(out)
        out = self.drop2(out)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs=1, channels=(32, 32, 64), kernel_size=3, dropout=0.2):
        super().__init__()
        layers=[]; in_ch=num_inputs
        for i, ch in enumerate(channels):
            dilation = 2**i
            padding = (kernel_size-1)*dilation
            layers.append(TemporalBlock(in_ch, ch, kernel_size, stride=1, dilation=dilation, padding=padding, dropout=dropout))
            in_ch = ch
        self.network = nn.Sequential(*layers)
    def forward(self, x):
        # x: (B,L,1) -> permute to (B,C,L)
        return self.network(x.transpose(1,2)).transpose(1,2)

class TCNTransformer(nn.Module):
    """TCN encoder + Transformer decoder for forecasting."""
    def __init__(self, d_model=128, nhead=4, depth=2, pred_len=24):
        super().__init__()
        self.tcn = TCN(1, (32,64,128))
        self.proj = nn.Linear(128, d_model)
        enc = nn.TransformerEncoderLayer(d_model, nhead, d_model*4, 0.1, batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, depth)
        self.dec_in = nn.Linear(1, d_model)
        dec = nn.TransformerDecoderLayer(d_model, nhead, d_model*4, 0.1, batch_first=True, norm_first=True)
        self.decoder = nn.TransformerDecoder(dec, depth)
        self.head = nn.Linear(d_model, 1)
        self.pred_len = pred_len
    def forward(self, x_enc, x_dec):
        h = self.tcn(x_enc)
        h = self.encoder(self.proj(h))
        tgt = self.dec_in(x_dec)
        y = self.decoder(tgt, h)
        return self.head(y)
