
import torch
import torch.nn as nn
from typing import List, Dict
from nlp_ae_common import MLPBlock, TextAvgEmbed

def _resize_linear(old: nn.Linear, new_in: int, new_out: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out)
    with torch.no_grad():
        in_overlap = min(old.in_features, new_in)
        out_overlap = min(old.out_features, new_out)
        if in_overlap > 0 and out_overlap > 0:
            new.weight[:out_overlap, :in_overlap].copy_(old.weight[:out_overlap, :in_overlap])
            if old.bias is not None and new.bias is not None:
                new.bias[:out_overlap].copy_(old.bias[:out_overlap])
    return new

def _resize_bn1d(old: nn.BatchNorm1d, new_features: int) -> nn.BatchNorm1d:
    new = nn.BatchNorm1d(new_features)
    with torch.no_grad():
        overlap = min(old.num_features, new_features)
        if overlap > 0:
            new.weight[:overlap].copy_(old.weight[:overlap])
            new.bias[:overlap].copy_(old.bias[:overlap])
            new.running_mean[:overlap].copy_(old.running_mean[:overlap])
            new.running_var[:overlap].copy_(old.running_var[:overlap])
    return new

class AdaptiveTextAE(nn.Module):
    def __init__(self, vocab_size: int, emb_dim: int, hidden: List[int], rep_dim: int, use_bn: bool=True):
        super().__init__()
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden = list(hidden)
        self.rep_dim = rep_dim
        self.use_bn = use_bn
        self._build_modules()
        self.global_epoch = 0

    def _build_modules(self):
        self.encoder_tok = TextAvgEmbed(self.vocab_size, self.emb_dim)
        layers, prev = [], self.emb_dim
        for w in self.hidden:
            layers.append(MLPBlock(prev, w, self.use_bn)); prev = w
        self.backbone = nn.Sequential(*layers)
        self.rep = nn.Linear(prev, self.rep_dim)
        self.decoder = nn.Linear(self.rep_dim, self.vocab_size)

    def forward(self, view):
        tok, lens = view
        h0 = self.encoder_tok(tok, lens)
        h = self.backbone(h0) if len(self.backbone) > 0 else h0
        z = self.rep(h)
        logits = self.decoder(z)
        return logits

    def total_neurons(self): return sum(self.hidden) + self.rep_dim
    def depth(self): return len(self.hidden) + 1

    def snapshot(self) -> Dict:
        return {"state": {k: v.detach().cpu() for k, v in self.state_dict().items()}, "hidden": list(self.hidden), "rep_dim": int(self.rep_dim)}
    def restore(self, snap: Dict):
        self.hidden = list(snap["hidden"]); self.rep_dim = int(snap["rep_dim"])
        self._build_modules(); self.load_state_dict(snap["state"], strict=True)

    def append_depth(self):
        new_w = self.hidden[-1] if len(self.hidden) > 0 else max(256, self.rep_dim)
        self.hidden.append(int(new_w))
        old = {"backbone": self.backbone, "rep": self.rep, "decoder": self.decoder}
        self._build_modules()
        for i, (old_blk, new_blk) in enumerate(zip(old["backbone"], self.backbone)):
            new_blk.linear = _resize_linear(old_blk.linear, new_blk.linear.in_features, new_blk.linear.out_features)
            if old_blk.bn is not None and new_blk.bn is not None:
                new_blk.bn = _resize_bn1d(old_blk.bn, new_blk.bn.num_features)
            self.backbone[i] = new_blk
        self.rep = _resize_linear(old["rep"], self.rep.in_features, self.rep.out_features)
        self.decoder = _resize_linear(old["decoder"], self.decoder.in_features, self.decoder.out_features)

    def widen_all(self, ex_k: int):
        self.hidden = [int(w + ex_k) for w in self.hidden]
        self.rep_dim = int(self.rep_dim + ex_k)
        old = {"backbone": self.backbone, "rep": self.rep, "decoder": self.decoder}
        self._build_modules()
        for i, (old_blk, new_blk) in enumerate(zip(old["backbone"], self.backbone)):
            new_blk.linear = _resize_linear(old_blk.linear, new_blk.linear.in_features, new_blk.linear.out_features)
            if old_blk.bn is not None and new_blk.bn is not None:
                new_blk.bn = _resize_bn1d(old_blk.bn, new_blk.bn.num_features)
            self.backbone[i] = new_blk
        self.rep = _resize_linear(old["rep"], self.rep.in_features, self.rep.out_features)
        self.decoder = _resize_linear(old["decoder"], self.decoder.in_features, self.decoder.out_features)
