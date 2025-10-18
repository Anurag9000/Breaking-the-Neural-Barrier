
from dataclasses import dataclass
import torch, torch.nn as nn, torch.nn.functional as F
from dnn_ae_graph import load_planetoid
from adp_ae_resize import _resize_linear

class AdaptiveDNNNodeAE(nn.Module):
    """Autoencoder with width/depth growth; node-as-neuron input."""
    def __init__(self, num_nodes: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        assert depth>=1
        self.num_nodes=num_nodes; self.hidden=hidden; self.depth=depth
        self.feature_proj=None; self.decoder_out=None
        self.in_lin=nn.Linear(num_nodes, hidden, bias=False)
        self.hiddens=nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(depth-1)])
        self.reset_parameters()
    def reset_parameters(self):
        nn.init.kaiming_normal_(self.in_lin.weight, nonlinearity="relu")
        for lin in self.hiddens: nn.init.kaiming_normal_(lin.weight, nonlinearity="relu")
    def _ensure_feature_proj(self, F):
        if self.feature_proj is None:
            self.feature_proj=nn.Parameter(torch.randn(F)*0.02)
    def _ensure_decoder(self, F):
        if self.decoder_out is None:
            self.decoder_out=nn.Linear(self.hidden, self.num_nodes*F, bias=True)
            nn.init.zeros_(self.decoder_out.bias); nn.init.kaiming_normal_(self.decoder_out.weight, nonlinearity="linear")
    def forward(self, X):
        N,F=X.shape; assert N==self.num_nodes
        self._ensure_feature_proj(F); self._ensure_decoder(F)
        z=(X@self.feature_proj).unsqueeze(0)
        z=torch.relu(self.in_lin(z))
        for lin in self.hiddens: z=torch.relu(lin(z))
        return self.decoder_out(z).view(N,F)
    @torch.no_grad()
    def total_neurons(self, F): return self.hidden*(self.depth+1)+self.num_nodes*F
    # mutations
    def widen_all(self, delta: int, F: int):
        new_h=self.hidden+delta
        self.in_lin = _resize_linear(self.in_lin, self.num_nodes, new_h)
        self.hiddens = nn.ModuleList([_resize_linear(l, new_h, new_h) for l in self.hiddens])
        if self.decoder_out is not None:
            self.decoder_out = _resize_linear(self.decoder_out, new_h, self.num_nodes*F)
        self.hidden=new_h
    def append_depth(self):
        new_lin=nn.Linear(self.hidden, self.hidden, bias=False)
        nn.init.kaiming_normal_(new_lin.weight, nonlinearity="relu")
        self.hiddens.append(new_lin); self.depth+=1
    def snapshot(self): return {k:v.detach().cpu().clone() for k,v in self.state_dict().items()}
    def restore(self, st): self.load_state_dict(st, strict=True)

@dataclass
class TrainCfg:
    lr:float=1e-3; weight_decay:float=5e-4; max_epochs:int=1000; patience:int=100; grad_clip:float=1.0
    device:str="cuda" if torch.cuda.is_available() else "cpu"

def train_early_stop(model: AdaptiveDNNNodeAE, data, cfg: TrainCfg):
    X,y,tr,val,te = data
    X=X.to(cfg.device); tr,val,te=[m.to(cfg.device) for m in (tr,val,te)]
    model=model.to(cfg.device); opt=torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best=float('inf'); best_state=None; pat=cfg.patience
    for _ in range(cfg.max_epochs):
        model.train(); opt.zero_grad(set_to_none=True)
        Xh=model(X); loss=F.mse_loss(Xh[tr], X[tr]); loss.backward()
        if cfg.grad_clip: torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        model.eval()
        with torch.no_grad():
            v=F.mse_loss(model(X)[val], X[val]).item()
        if v<best-1e-9: best=v; best_state=model.snapshot(); pat=cfg.patience
        else: pat-=1
        if pat<=0: break
    if best_state: model.restore(best_state)
    with torch.no_grad():
        test=F.mse_loss(model(X)[te], X[te]).item()
    return best, test, None
