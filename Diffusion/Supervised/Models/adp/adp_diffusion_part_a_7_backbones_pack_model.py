# ============================================================
# File: adp_diff_backbones.py  (MODEL)
# Single-model DDPM (ε-pred) with a plug-in denoiser backbone:
#   --backbone {unet,resnet,convnext,vit,dit,swin,unetpp,tiny}
# Each backbone implements the ADP API so all 6 ADP policies work.
# No EMA/teacher. Lightweight reference implementations.
# ============================================================

from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# Diffusion cosine schedule
# -----------------------------

def cosine_beta_schedule(T: int, s: float = 0.008) -> torch.Tensor:
    steps = T + 1
    x = torch.linspace(0, T, steps)
    a_bar = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
    a_bar = a_bar / a_bar[0]
    betas = 1 - (a_bar[1:] / a_bar[:-1])
    return torch.clip(betas, 1e-6, 0.999)

# ============================================================
# Backbones: all implement a common ADP interface
#   • forward(x, t_norm) → eps_pred (same shape as x)
#   • append_depth(), widen_all(ex_k), neurons(), snapshot_state(), restore_state()
# ============================================================

class TimeMLP(nn.Module):
    def __init__(self, time_ch=128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, time_ch), nn.SiLU(), nn.Linear(time_ch, time_ch))
        self.time_ch = time_ch
    def forward(self, t: torch.Tensor):
        if t.dim() == 1:
            t = t[:, None]
        return self.net(t)
    def expand(self, emb: torch.Tensor, H: int, W: int):
        B = emb.size(0)
        return emb[:, :, None, None].expand(B, emb.size(1), H, W)

class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=nn.SiLU):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, s, p)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = act(inplace=True)
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

# ----- UNet (baseline) -----
class UNetAdaptive(nn.Module):
    def __init__(self, in_ch: int, widths: List[int], time_ch: int = 128, dense_skips: bool = False):
        super().__init__()
        self.widths = list(widths)
        self.time = TimeMLP(time_ch)
        self.dense_skips = dense_skips  # if True, emulate UNet++ dense skip concatenations
        self._build(in_ch)
    def _build(self, in_ch):
        w = self.widths; T = self.time.time_ch
        self.num = len(w)
        self.down, self.pool = nn.ModuleList([]), nn.ModuleList([])
        ch = in_ch
        for wi in w:
            self.down += [ConvBNAct(ch + T, wi), ConvBNAct(wi, wi)]
            self.pool += [nn.AvgPool2d(2)]
            ch = wi
        self.mid1, self.mid2 = ConvBNAct(ch + T, ch), ConvBNAct(ch, ch)
        self.up, self.upx = nn.ModuleList([]), nn.ModuleList([])
        up_ch = ch
        for wi in reversed(w):
            self.upx += [nn.ConvTranspose2d(up_ch, up_ch, 4, 2, 1)]
            skip_in = wi
            if self.dense_skips:
                skip_in = wi * 2  # coarse emulation: allow an extra skip concat channel budget
            self.up += [ConvBNAct(up_ch + skip_in + T, wi), ConvBNAct(wi, wi)]
            up_ch = wi
        self.head = nn.Conv2d(up_ch, in_ch, 1)
    def _inj(self, x, t_emb):
        B, C, H, W = x.shape
        return torch.cat([x, self.time.expand(t_emb, H, W)], dim=1)
    def forward(self, x, t):
        t_emb = self.time(t)
        feats = []
        cur = x
        for i in range(self.num):
            cur = self.down[2*i](self._inj(cur, t_emb)); cur = self.down[2*i+1](cur)
            feats.append(cur); cur = self.pool[i](cur)
        cur = self.mid1(self._inj(cur, t_emb)); cur = self.mid2(cur)
        for i in range(self.num):
            skip = feats[-(i+1)]
            if self.dense_skips:
                # simple dense skip: concat with previous up output if available
                if i > 0:
                    skip = torch.cat([skip, feats[-i]], dim=1)
            cur = self.upx[i](cur)
            if cur.size(-1) != skip.size(-1):
                cur = F.interpolate(cur, size=skip.shape[-2:], mode='nearest')
            cur = torch.cat([cur, skip], dim=1)
            cur = self.up[2*i](self._inj(cur, t_emb)); cur = self.up[2*i+1](cur)
        return self.head(cur)
    # ADP
    def neurons(self):
        return int(sum(self.widths))
    def snapshot_state(self):
        return {k: v.detach().clone() for k,v in self.state_dict().items()}
    def restore_state(self, s):
        self.load_state_dict(s, strict=True)
    def append_depth(self):
        self.widths.append(self.widths[-1]); self._build(self.head.in_channels)
    def widen_all(self, ex_k):
        old = self.state_dict(); self.widths=[w+ex_k for w in self.widths]; self._build(self.head.in_channels)
        new = self.state_dict()
        for k in new:
            if k in old:
                src,dst=old[k],new[k]; com=tuple(min(a,b) for a,b in zip(src.shape,dst.shape)); sl=tuple(slice(0,c) for c in com); dst[sl]=src[sl]
        self.load_state_dict(new, strict=False)

# ----- ResNet denoiser -----
class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.c1=ConvBNAct(ch, ch); self.c2=ConvBNAct(ch, ch)
    def forward(self,x):
        return x + self.c2(self.c1(x))

class ResNetAdaptive(nn.Module):
    def __init__(self, in_ch:int, widths:List[int], time_ch:int=128):
        super().__init__()
        self.widths=list(widths); self.time=TimeMLP(time_ch); self._build(in_ch)
    def _build(self,in_ch):
        w=self.widths; T=self.time.time_ch; self.num=len(w)
        layers=[]; ch=in_ch
        for wi in w:
            layers += [ConvBNAct(ch+T, wi), ResBlock(wi)]
            ch=wi
        self.stem=nn.ModuleList(layers)
        self.head=nn.Conv2d(ch,in_ch,1)
    def _inj(self,x,t_emb):
        B,C,H,W=x.shape; return torch.cat([x,self.time.expand(t_emb,H,W)],dim=1)
    def forward(self,x,t):
        te=self.time(t); cur=x
        for i in range(self.num):
            cur=self.stem[2*i](self._inj(cur,te)); cur=self.stem[2*i+1](cur)
        return self.head(cur)
    def neurons(self): return int(sum(self.widths))
    def snapshot_state(self): return {k:v.detach().clone() for k,v in self.state_dict().items()}
    def restore_state(self,s): self.load_state_dict(s, strict=True)
    def append_depth(self): self.widths.append(self.widths[-1]); self._build(self.head.in_channels)
    def widen_all(self,ex_k):
        old=self.state_dict(); self.widths=[w+ex_k for w in self.widths]; self._build(self.head.in_channels)
        new=self.state_dict();
        for k in new:
            if k in old:
                src,dst=old[k],new[k]; com=tuple(min(a,b) for a,b in zip(src.shape,dst.shape)); sl=tuple(slice(0,c) for c in com); dst[sl]=src[sl]
        self.load_state_dict(new, strict=False)

# ----- ConvNeXt-like denoiser -----
class ConvNeXtBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.dw = nn.Conv2d(ch, ch, 7, padding=3, groups=ch)
        self.norm = nn.LayerNorm(ch)
        self.pw1 = nn.Linear(ch, 4*ch)
        self.act = nn.GELU()
        self.pw2 = nn.Linear(4*ch, ch)
    def forward(self, x):
        r = x
        x = self.dw(x)
        # permute to (B,H,W,C) for LayerNorm/Linear then back
        x = x.permute(0,2,3,1)
        x = self.norm(x)
        x = self.pw2(self.act(self.pw1(x)))
        x = x.permute(0,3,1,2)
        return r + x

class ConvNeXtAdaptive(nn.Module):
    def __init__(self, in_ch:int, widths:List[int], time_ch:int=128):
        super().__init__(); self.widths=list(widths); self.time=TimeMLP(time_ch); self._build(in_ch)
    def _build(self,in_ch):
        w=self.widths; T=self.time.time_ch; self.num=len(w)
        blocks=[]; ch=in_ch
        for wi in w:
            blocks += [ConvBNAct(ch+T, wi), ConvNeXtBlock(wi)]
            ch=wi
        self.blocks=nn.ModuleList(blocks); self.head=nn.Conv2d(ch,in_ch,1)
    def _inj(self,x,t_emb):
        B,C,H,W=x.shape; return torch.cat([x,self.time.expand(t_emb,H,W)],dim=1)
    def forward(self,x,t):
        te=self.time(t); cur=x
        for i in range(self.num):
            cur=self.blocks[2*i](self._inj(cur,te)); cur=self.blocks[2*i+1](cur)
        return self.head(cur)
    def neurons(self): return int(sum(self.widths))
    def snapshot_state(self): return {k:v.detach().clone() for k,v in self.state_dict().items()}
    def restore_state(self,s): self.load_state_dict(s, strict=True)
    def append_depth(self): self.widths.append(self.widths[-1]); self._build(self.head.in_channels)
    def widen_all(self,ex_k):
        old=self.state_dict(); self.widths=[w+ex_k for w in self.widths]; self._build(self.head.in_channels)
        new=self.state_dict();
        for k in new:
            if k in old:
                src,dst=old[k],new[k]; com=tuple(min(a,b) for a,b in zip(src.shape,dst.shape)); sl=tuple(slice(0,c) for c in com); dst[sl]=src[sl]
        self.load_state_dict(new, strict=False)

# ----- Token/Transformer-style (ViT/DiT/Swin-lite) -----
class PatchEmbed(nn.Module):
    def __init__(self, in_ch=3, embed=192, patch=4):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, embed, kernel_size=patch, stride=patch)
    def forward(self, x):
        x = self.proj(x)  # (B, C, H/P, W/P)
        B,C,H,W=x.shape
        return x.flatten(2).transpose(1,2), (H,W)  # (B, HW, C), grid

class PatchUnembed(nn.Module):
    def __init__(self, out_ch=3, embed=192, patch=4):
        super().__init__()
        self.out_ch = out_ch; self.embed=embed; self.patch=patch
        self.proj = nn.ConvTranspose2d(embed, out_ch, kernel_size=patch, stride=patch)
    def forward(self, tokens, grid):
        B,N,C=tokens.shape; H,W=grid
        x = tokens.transpose(1,2).reshape(B,C,H,W)
        return self.proj(x)

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads=4, mlp_ratio=4.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim); self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.ln2 = nn.LayerNorm(dim); self.mlp = nn.Sequential(nn.Linear(dim, int(dim*mlp_ratio)), nn.GELU(), nn.Linear(int(dim*mlp_ratio), dim))
    def forward(self, x):
        h = x; x = self.ln1(x); x = self.attn(x, x, x, need_weights=False)[0]; x = x + h
        h = x; x = self.ln2(x); x = self.mlp(x); return x + h

class TokenAdaptive(nn.Module):
    """Generic token model; --backbone differentiates via minor options (vit/dit/swin).
    ADP width = embed dim; depth = #blocks.
    """
    def __init__(self, in_ch:int, embed:int=192, depth:int=6, time_ch:int=128, patch:int=4, flavor:str='vit'):
        super().__init__()
        self.embed = embed; self.depth = depth; self.flavor = flavor
        self.time = TimeMLP(time_ch)
        self.patch = patch
        self.patch_embed = PatchEmbed(in_ch, embed, patch)
        self.blocks = nn.ModuleList([TransformerBlock(embed, heads=4 if flavor!='dit' else 8) for _ in range(depth)])
        self.unembed = PatchUnembed(out_ch=in_ch, embed=embed, patch=patch)
        self.out_conv = nn.Conv2d(in_channels=in_ch, out_channels=in_ch, kernel_size=1)
    def forward(self, x, t):
        B,C,H,W=x.shape; te=self.time(t)
        # inject time by adding a global token bias
        tokens, grid = self.patch_embed(x)
        # broadcast time into token space
        tvec = self.time.expand(te, 1, 1)[:, :, 0, 0]  # (B, time_ch)
        # project tvec to embed
        tproj = F.linear(tvec, torch.zeros(self.embed, tvec.size(1), device=x.device), bias=torch.zeros(self.embed, device=x.device))
        tokens = tokens + tproj[:, None, :]
        for blk in self.blocks:
            tokens = blk(tokens)
        x_hat = self.unembed(tokens, grid)
        return self.out_conv(x_hat)
    # ADP ops
    def neurons(self): return int(self.embed * self.depth)
    def snapshot_state(self): return {k:v.detach().clone() for k,v in self.state_dict().items()}
    def restore_state(self, s): self.load_state_dict(s, strict=True)
    def append_depth(self):
        self.depth += 1; self.blocks.append(TransformerBlock(self.embed, heads=8 if self.flavor=='dit' else 4))
    def widen_all(self, ex_k:int):
        new_dim = self.embed + ex_k
        # rebuild blocks with overlap-copy
        old = self.state_dict();
        self.embed = new_dim
        self.patch_embed = PatchEmbed(in_ch=self.unembed.out_ch, embed=new_dim, patch=self.patch)
        self.blocks = nn.ModuleList([TransformerBlock(new_dim, heads=8 if self.flavor=='dit' else 4) for _ in range(self.depth)])
        self.unembed = PatchUnembed(out_ch=self.unembed.out_ch, embed=new_dim, patch=self.patch)
        self.out_conv = nn.Conv2d(self.unembed.out_ch, self.unembed.out_ch, 1)
        new = self.state_dict()
        for k in new:
            if k in old:
                src,dst=old[k],new[k]; com=tuple(min(a,b) for a,b in zip(src.shape,dst.shape)); sl=tuple(slice(0,c) for c in com); dst[sl]=src[sl]
        self.load_state_dict(new, strict=False)

# Factories for backbones

def make_backbone(name:str, in_ch:int, widths:List[int], time_ch:int=128):
    name = name.lower()
    if name == 'unet':
        return UNetAdaptive(in_ch, widths, time_ch=time_ch, dense_skips=False)
    if name == 'unetpp':
        return UNetAdaptive(in_ch, widths, time_ch=time_ch, dense_skips=True)
    if name == 'resnet':
        return ResNetAdaptive(in_ch, widths, time_ch=time_ch)
    if name == 'convnext':
        return ConvNeXtAdaptive(in_ch, widths, time_ch=time_ch)
    if name in ('vit','dit','swin','tiny'):
        # map widths to (embed, depth); widths[0]=embed, len(widths)=depth baseline
        embed = widths[0]
        depth = max(1, len(widths))
        flavor = 'vit' if name=='vit' else ('dit' if name=='dit' else ('swin' if name=='swin' else 'tiny'))
        return TokenAdaptive(in_ch, embed=embed, depth=depth, time_ch=time_ch, patch=4 if name!='tiny' else 8, flavor=flavor)
    raise ValueError(f'Unknown backbone {name}')

# ============================================================
# Single-model DDPM core (ε-pred), agnostic to backbone
# ============================================================

class EpsDDPMSingleModel(nn.Module):
    def __init__(self, img_ch:int=3, widths:List[int]=[32,64,96], T:int=1000, backbone:str='unet'):
        super().__init__()
        self.T=int(T)
        self.register_buffer('betas', cosine_beta_schedule(T))
        self.register_buffer('alphas', 1.0 - self.betas)
        self.register_buffer('alphas_cumprod', torch.cumprod(self.alphas, dim=0))
        self.net = make_backbone(backbone, img_ch, widths, time_ch=128)
    def q_sample(self, x0:torch.Tensor, t_idx:torch.Tensor, eps:Optional[torch.Tensor]=None):
        if eps is None: eps=torch.randn_like(x0)
        a_bar=self.alphas_cumprod[t_idx][:,None,None,None]
        x_t = torch.sqrt(a_bar)*x0 + torch.sqrt(1.0-a_bar)*eps
        return x_t, eps
    def forward(self, x:torch.Tensor)->torch.Tensor:
        B=x.size(0); dev=x.device
        t_idx=torch.randint(0, self.T, (B,), device=dev, dtype=torch.long)
        t_norm=(t_idx.float()+0.5)/self.T
        x_t, eps = self.q_sample(x, t_idx)
        eps_pred = self.net(x_t, t_norm)
        return 0.5*F.mse_loss(eps_pred, eps)
    @torch.no_grad()
    def sample(self, B:int, img_ch:int, H:int, W:int, steps:Optional[int]=None, device:Optional[torch.device]=None):
        if device is None: device=next(self.parameters()).device
        if steps is None: steps=self.T
        x=torch.randn(B,img_ch,H,W,device=device)
        for i in reversed(range(steps)):
            t=torch.full((B,), i, device=device, dtype=torch.long)
            t_norm=(t.float()+0.5)/self.T
            eps=self.net(x,t_norm)
            beta=self.betas[i]; alpha=self.alphas[i]; a_bar=self.alphas_cumprod[i]
            noise=torch.randn_like(x) if i>0 else torch.zeros_like(x)
            x = (1.0/torch.sqrt(alpha))*(x - beta/torch.sqrt(1.0-a_bar)*eps) + torch.sqrt(beta)*noise
        return torch.clamp(x, -1, 1)
    # ADP passthrough
    def neurons(self): return int(self.net.neurons())
    def snapshot_state(self): return {'net': self.net.state_dict()}
    def restore_state(self,snap): self.net.load_state_dict(snap['net'])
    def append_depth(self): self.net.append_depth()
    def widen_all(self, ex_k:int): self.net.widen_all(ex_k)

# ============================================================
# ES trainer and ADP policies (shared across backbones)
# ============================================================

@dataclass
class TrainCfg:
    lr: float = 2e-4
    max_epochs: int = 50
    es_patience: int = 10
    grad_clip: Optional[float] = 1.0
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'


def train_one_model(model:EpsDDPMSingleModel, train_loader, val_loader, cfg:TrainCfg)->Tuple[float,Dict]:
    device=torch.device(cfg.device); model.to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=cfg.lr)
    best=float('inf'); best_snap=None; bad=0
    for epoch in range(cfg.max_epochs):
        model.train()
        for x,_ in train_loader:
            x=x.to(device); loss=model(x)
            opt.zero_grad(set_to_none=True); loss.backward()
            if cfg.grad_clip: nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            opt.step()
        model.eval(); tot=0.0; n=0
        with torch.no_grad():
            for x,_ in val_loader:
                x=x.to(device); l=model(x); tot+=float(l.item())*x.size(0); n+=x.size(0)
        val=tot/max(1,n)
        if val+1e-8<best:
            best=val; best_snap={'model': model.state_dict()}; bad=0
        else: bad+=1
        if bad>=cfg.es_patience: break
    if best_snap is not None: model.load_state_dict(best_snap['model'])
    return best, best_snap

@dataclass
class SearchCfg:
    delta: float = 0.0
    trials_width: int = 50
    trials_depth: int = 50
    ex_k: int = 8
    max_neurons: Optional[int] = None

def _accept(v_improved:float, v_base:float, d:float)->bool: return v_improved < (v_base - d)

def adp_depth_then_width(model, train_loader, val_loader, tr, s:SearchCfg):
    base,_=train_one_model(model, train_loader, val_loader, tr)
    d_fail=0
    for _ in range(s.trials_depth):
        pre=model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; d_fail=0
        else: model.restore_state(pre); d_fail+=1; if d_fail>=2: break
    w_fail=0
    for _ in range(s.trials_width):
        pre=model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; w_fail=0
        else: model.restore_state(pre); w_fail+=1; if w_fail>=2: break
    return base

def adp_width_then_depth(model, train_loader, val_loader, tr, s:SearchCfg):
    base,_=train_one_model(model, train_loader, val_loader, tr)
    w_fail=0
    for _ in range(s.trials_width):
        pre=model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; w_fail=0
        else: model.restore_state(pre); w_fail+=1; if w_fail>=2: break
    d_fail=0
    for _ in range(s.trials_depth):
        pre=model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; d_fail=0
        else: model.restore_state(pre); d_fail+=1; if d_fail>=2: break
    return base

def adp_alt_depth_first(model, train_loader, val_loader, tr, s:SearchCfg):
    base,_=train_one_model(model, train_loader, val_loader, tr)
    while True:
        improved=False
        pre=model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; improved=True
        else: model.restore_state(pre)
        pre=model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; improved=True
        else: model.restore_state(pre)
        if not improved: break
    return base

def adp_alt_width_first(model, train_loader, val_loader, tr, s:SearchCfg):
    base,_=train_one_model(model, train_loader, val_loader, tr)
    while True:
        improved=False
        pre=model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; improved=True
        else: model.restore_state(pre)
        pre=model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; improved=True
        else: model.restore_state(pre)
        if not improved: break
    return base

def adp_depth_only(model, train_loader, val_loader, tr, s:SearchCfg):
    base,_=train_one_model(model, train_loader, val_loader, tr)
    fail=0
    for _ in range(s.trials_depth):
        pre=model.snapshot_state(); model.append_depth()
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; fail=0
        else: model.restore_state(pre); fail+=1; if fail>=2: break
    return base

def adp_width_only(model, train_loader, val_loader, tr, s:SearchCfg):
    base,_=train_one_model(model, train_loader, val_loader, tr)
    fail=0
    for _ in range(s.trials_width):
        pre=model.snapshot_state(); model.widen_all(s.ex_k)
        if s.max_neurons and model.neurons()>s.max_neurons: model.restore_state(pre); break
        v,_=train_one_model(model, train_loader, val_loader, tr)
        if _accept(v,base,s.delta): base=v; fail=0
        else: model.restore_state(pre); fail+=1; if fail>=2: break
    return base

POLICIES={
    'depth2width': adp_depth_then_width,
    'width2depth': adp_width_then_depth,
    'alt_depth': adp_alt_depth_first,
    'alt_width': adp_alt_width_first,
    'depth_only': adp_depth_only,
    'width_only': adp_width_only,
}

# ============================================================
# End of adp_diff_backbones.py
# ============================================================
