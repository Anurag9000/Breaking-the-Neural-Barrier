import torch
import torch.nn as nn
import torch.nn.functional as F

class NPairGRU(nn.Module):
    def __init__(self, input_dim:int, hidden_dim:int, proj_dim:int=128, num_layers:int=1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.proj = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU(inplace=True), nn.Linear(hidden_dim, proj_dim))
    def encode(self, x):
        h,_=self.encoder(x); h=h.mean(dim=1)
        z=F.normalize(self.proj(h),dim=-1)
        return z

if __name__=='__main__':
    B,T,D=8,64,16
    net=NPairGRU(D,128,64)
    x=torch.randn(B,T,D)
    print(net.encode(x).shape)
