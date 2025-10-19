import torch
import torch.nn as nn
import torch.nn.functional as F

class VRAEGRU(nn.Module):
    """Variational Recurrent Autoencoder with GRU encoder/decoder.
    Single-model; reparameterized latent z -> GRU decoder to reconstruct sequence.
    """
    def __init__(self, input_dim:int, hidden_dim:int, latent_dim:int, num_layers:int=1):
        super().__init__()
        self.encoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.latent_to_h0 = nn.Linear(latent_dim, hidden_dim)
        self.decoder = nn.GRU(input_dim, hidden_dim, num_layers=num_layers, batch_first=True)
        self.head = nn.Linear(hidden_dim, input_dim)
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim

    def encode(self, x):
        h,_ = self.encoder(x)           # (B,T,H)
        h_last = h[:,-1,:]              # (B,H)
        mu, logvar = self.mu(h_last), self.logvar(h_last)
        return mu, logvar

    @staticmethod
    def reparameterize(mu, logvar):
        std = (0.5*logvar).exp()
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, dec_inp):
        h0 = self.latent_to_h0(z).unsqueeze(0).repeat(self.num_layers,1,1)  # (L,B,H)
        out,_ = self.decoder(dec_inp, h0)
        return self.head(out)

    def forward(self, x, dec_inp):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, dec_inp)
        return recon, mu, logvar

    @staticmethod
    def kld(mu, logvar):
        # KL(N(mu,sigma)||N(0,1))
        return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

if __name__ == '__main__':
    B,T,D=4,16,8
    net=VRAEGRU(D,64,32)
    x=torch.randn(B,T,D)
    dec_inp=torch.zeros_like(x); dec_inp[:,1:,:]=x[:,:-1,:]
    y,mu,lv=net(x,dec_inp)
    print(y.shape, mu.shape)
