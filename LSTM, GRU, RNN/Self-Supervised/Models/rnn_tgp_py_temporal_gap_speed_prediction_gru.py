import torch
import torch.nn as nn

class TGPGRU(nn.Module):
    """
    Predict temporal gap / speed category between sequence segments.
    Binary or multi-class classification depending on runner labels.
    """
    def __init__(self, input_dim:int, hidden_dim:int, num_layers:int=1, num_classes:int=3):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(
            nn.Linear(2*hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, num_classes)
        )
    def forward(self, x):
        h,_ = self.encoder(x)
        h = h.mean(dim=1)
        return self.head(h)

if __name__=='__main__':
    B,T,D=8,64,16
    net=TGPGRU(D,64,num_classes=3)
    x=torch.randn(B,T,D)
    print(net(x).shape)
