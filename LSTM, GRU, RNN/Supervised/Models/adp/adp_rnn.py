
import math
import time
from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

# ==============================
# Utilities: overlap-copy resize
# ==============================

def _copy_param_slice_(new_param: torch.nn.Parameter, old_param: torch.nn.Parameter):
    with torch.no_grad():
        new_param.zero_()
        # Copy overlapping slice
        common_shape = tuple(min(a, b) for a, b in zip(new_param.shape, old_param.shape))
        if len(common_shape) == 0:
            return
        slices = tuple(slice(0, s) for s in common_shape)
        new_param[slices].copy_(old_param[slices])


def _resize_linear(in_features: int, out_features: int, old: nn.Linear) -> nn.Linear:
    new = nn.Linear(in_features, out_features, bias=(old.bias is not None))
    _copy_param_slice_(new.weight, old.weight)
    if old.bias is not None:
        _copy_param_slice_(new.bias, old.bias)
    return new


def _resize_embedding(num_embeddings: int, embedding_dim: int, old: nn.Embedding) -> nn.Embedding:
    new = nn.Embedding(num_embeddings, embedding_dim, padding_idx=old.padding_idx)
    _copy_param_slice_(new.weight, old.weight)
    return new


def _make_rnn(input_size: int, hidden_size: int, num_layers: int, bidirectional: bool=False, dropout: float=0.0, nonlinearity: str="tanh") -> nn.RNN:
    return nn.RNN(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        nonlinearity=nonlinearity,  # 'tanh' or 'relu'
        bias=True,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
        bidirectional=bidirectional
    )


def _resize_rnn_hidden(old: nn.RNN, new_hidden: int) -> nn.RNN:
    # Return a new RNN with resized hidden_size, overlap-copying the weights/biases.
    new = _make_rnn(
        input_size=old.input_size,
        hidden_size=new_hidden,
        num_layers=old.num_layers,
        bidirectional=old.bidirectional,
        dropout=old.dropout,
        nonlinearity=old.nonlinearity if hasattr(old, 'nonlinearity') else 'tanh'
    )
    # Copy all layers directions
    num_dirs = 2 if old.bidirectional else 1
    with torch.no_grad():
        for layer in range(old.num_layers):
            for d in range(num_dirs):
                suffix = f"_l{layer}"
                if d == 1:
                    suffix += "_reverse"
                # weight_ih / hh and biases
                _copy_param_slice_(getattr(new, f"weight_ih{suffix}"), getattr(old, f"weight_ih{suffix}"))
                _copy_param_slice_(getattr(new, f"weight_hh{suffix}"), getattr(old, f"weight_hh{suffix}"))
                _copy_param_slice_(getattr(new, f"bias_ih{suffix}"), getattr(old, f"bias_ih{suffix}"))
                _copy_param_slice_(getattr(new, f"bias_hh{suffix}"), getattr(old, f"bias_hh{suffix}"))
    return new


def _append_rnn_layer(old: nn.RNN) -> nn.RNN:
    # Return a new RNN with num_layers+1, overlap-copying existing layers and
    # initializing the last layer from the previous top layer (best-effort).
    new = _make_rnn(
        input_size=old.input_size,
        hidden_size=old.hidden_size,
        num_layers=old.num_layers + 1,
        bidirectional=old.bidirectional,
        dropout=old.dropout,
        nonlinearity=old.nonlinearity if hasattr(old, 'nonlinearity') else 'tanh'
    )
    num_dirs = 2 if old.bidirectional else 1
    with torch.no_grad():
        # copy existing layers
        for layer in range(old.num_layers):
            for d in range(num_dirs):
                suffix = f"_l{layer}"
                if d == 1:
                    suffix += "_reverse"
                for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                    _copy_param_slice_(getattr(new, f"{name}{suffix}"), getattr(old, f"{name}{suffix}"))
        # init last layer from previous top layer
        prev = old.num_layers - 1
        for d in range(num_dirs):
            src_suffix = f"_l{prev}"
            dst_suffix = f"_l{old.num_layers}"
            if d == 1:
                src_suffix += "_reverse"
                dst_suffix += "_reverse"
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                _copy_param_slice_(getattr(new, f"{name}{dst_suffix}"), getattr(old, f"{name}{src_suffix}"))
    return new


# ==============================
# Model
# ==============================

class TextRNN(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        num_classes: int,
        emb_dim: int = 128,
        hidden_size: int = 128,
        num_layers: int = 1,
        bidirectional: bool = False,
        dropout: float = 0.1,
        nonlinearity: str = "tanh",
        pad_idx: int = 0,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.num_classes = num_classes
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.dropout_p = dropout
        self.nonlinearity = nonlinearity
        self.pad_idx = pad_idx

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.rnn = _make_rnn(emb_dim, hidden_size, num_layers, bidirectional, dropout, nonlinearity)
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(out_dim, num_classes)

    # ----- growth ops -----
    def widen_hidden(self, delta: int):
        self.hidden_size += delta
        self.rnn = _resize_rnn_hidden(self.rnn, self.hidden_size)
        out_dim = self.hidden_size * (2 if self.bidirectional else 1)
        self.fc = _resize_linear(out_dim, self.num_classes, self.fc)

    def append_layer(self):
        self.num_layers += 1
        self.rnn = _append_rnn_layer(self.rnn)

    # ----- snapshot/restore -----
    def snapshot(self) -> Dict:
        return {
            "state": self.state_dict(),
            "arch": {
                "vocab_size": self.vocab_size,
                "num_classes": self.num_classes,
                "emb_dim": self.emb_dim,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "bidirectional": self.bidirectional,
                "dropout": self.dropout_p,
                "nonlinearity": self.nonlinearity,
                "pad_idx": self.pad_idx,
            }
        }

    @staticmethod
    def from_snapshot(snap: Dict) -> "TextRNN":
        arch = snap["arch"]
        model = TextRNN(**arch)
        model.load_state_dict(snap["state"])
        return model

    def forward(self, x, lengths=None):
        # x: (B, T) token ids
        emb = self.embedding(x)  # (B, T, E)
        if lengths is not None:
            packed = nn.utils.rnn.pack_padded_sequence(emb, lengths.cpu(), batch_first=True, enforce_sorted=False)
            out_packed, h_n = self.rnn(packed)
        else:
            out, h_n = self.rnn(emb)
        # Take last layer's hidden state (both directions if bi)
        if self.bidirectional:
            # h_n: (num_layers*2, B, H)
            last_fwd = h_n[-2]  # (B, H)
            last_bwd = h_n[-1]  # (B, H)
            feat = torch.cat([last_fwd, last_bwd], dim=-1)
        else:
            feat = h_n[-1]  # (B, H)
        feat = self.dropout(feat)
        logits = self.fc(feat)
        return logits


# ==============================
# Search configs
# ==============================

@dataclass
class TrainCfg:
    lr: float = 1e-3
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 200
    es_patience: int = 10


@dataclass
class SearchCfg:
    delta: float = 0.0
    ex_k: int = 32            # hidden-size widen step
    trials_width: int = 50
    trials_depth: int = 50
    max_total_epochs: int = 300
    max_layers: int = 12
    max_hidden: int = 2048


# ==============================
# Train / Eval
# ==============================

def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> Tuple[float, float]:
    model.eval()
    total = 0
    correct = 0
    total_loss = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for batch in loader:
            if isinstance(batch, dict):
                x, y = batch["x"].to(device), batch["y"].to(device)
                lengths = batch.get("lengths", None)
                if lengths is not None:
                    lengths = lengths.to(device)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                lengths = None
            logits = model(x, lengths)
            loss = criterion(logits, y)
            total_loss += loss.item() * y.size(0)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def train_until_plateau(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, cfg: TrainCfg, max_epochs: int) -> Tuple[float, Dict, int]:
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    criterion = nn.CrossEntropyLoss()
    best_snap = model.state_dict()
    best_val = float("inf")
    epochs_run = 0
    es_counter = 0
    for epoch in range(min(cfg.max_epochs, max_epochs)):
        model.train()
        for batch in train_loader:
            if isinstance(batch, dict):
                x, y = batch["x"].to(device), batch["y"].to(device)
                lengths = batch.get("lengths", None)
                if lengths is not None:
                    lengths = lengths.to(device)
            else:
                x, y = batch[0].to(device), batch[1].to(device)
                lengths = None
            logits = model(x, lengths)
            loss = criterion(logits, y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()

        val_loss, _ = evaluate(model, val_loader, device)
        epochs_run += 1
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es_counter = 0
        else:
            es_counter += 1
        if es_counter >= cfg.es_patience:
            break
    # load best
    model.load_state_dict(best_snap)
    return best_val, best_snap, epochs_run


# ==============================
# ADP search (six strategies via a single driver)
# ==============================

class ADP_RNN_Search:
    def __init__(self, model: TextRNN, train_cfg: TrainCfg, search_cfg: SearchCfg, device: Optional[torch.device]=None):
        self.model = model
        self.train_cfg = train_cfg
        self.search_cfg = search_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_epochs = 0

    def _accept(self, base_loss: float, prop_loss: float) -> bool:
        return prop_loss < (base_loss - self.search_cfg.delta)

    def _budget_left(self) -> int:
        return max(0, self.search_cfg.max_total_epochs - self.global_epochs)

    def _train_budgeted(self, model: TextRNN, train_loader, val_loader) -> float:
        remaining = self._budget_left()
        if remaining <= 0:
            return float("inf")
        val, snap, spent = train_until_plateau(model, train_loader, val_loader, self.device, self.train_cfg, remaining)
        self.global_epochs += spent
        return val

    # ---- strategy drivers ----

    def search_width_only(self, train_loader, val_loader):
        base_val = self._train_budgeted(self.model, train_loader, val_loader)
        failures = 0
        while self._budget_left() > 0 and failures < self.search_cfg.trials_width:
            base_snap = self.model.snapshot()
            if self.model.hidden_size + self.search_cfg.ex_k > self.search_cfg.max_hidden:
                break
            self.model.widen_hidden(self.search_cfg.ex_k)
            prop_val = self._train_budgeted(self.model, train_loader, val_loader)
            if self._accept(base_val, prop_val):
                base_val = prop_val
                failures = 0
            else:
                self.model = TextRNN.from_snapshot(base_snap)
                failures += 1
        return self.model

    def search_depth_only(self, train_loader, val_loader):
        base_val = self._train_budgeted(self.model, train_loader, val_loader)
        failures = 0
        while self._budget_left() > 0 and failures < self.search_cfg.trials_depth:
            base_snap = self.model.snapshot()
            if self.model.num_layers + 1 > self.search_cfg.max_layers:
                break
            self.model.append_layer()
            prop_val = self._train_budgeted(self.model, train_loader, val_loader)
            if self._accept(base_val, prop_val):
                base_val = prop_val
                failures = 0
            else:
                self.model = TextRNN.from_snapshot(base_snap)
                failures += 1
        return self.model

    def search_width_to_depth(self, train_loader, val_loader):
        base_val = self._train_budgeted(self.model, train_loader, val_loader)
        w_fails = 0
        while self._budget_left() > 0 and w_fails < self.search_cfg.trials_width:
            # propose width
            base_snap = self.model.snapshot()
            if self.model.hidden_size + self.search_cfg.ex_k > self.search_cfg.max_hidden:
                break
            self.model.widen_hidden(self.search_cfg.ex_k)
            prop_val = self._train_budgeted(self.model, train_loader, val_loader)
            if self._accept(base_val, prop_val):
                base_val = prop_val
                w_fails = 0
                # run depth mini-series
                d_fails = 0
                while self._budget_left() > 0 and d_fails < self.search_cfg.trials_depth:
                    series_snap = self.model.snapshot()
                    if self.model.num_layers + 1 > self.search_cfg.max_layers:
                        break
                    self.model.append_layer()
                    prop2_val = self._train_budgeted(self.model, train_loader, val_loader)
                    if self._accept(base_val, prop2_val):
                        base_val = prop2_val
                        d_fails = 0
                    else:
                        self.model = TextRNN.from_snapshot(series_snap)
                        d_fails += 1
            else:
                self.model = TextRNN.from_snapshot(base_snap)
                w_fails += 1
        return self.model

    def search_depth_to_width(self, train_loader, val_loader):
        base_val = self._train_budgeted(self.model, train_loader, val_loader)
        d_fails = 0
        while self._budget_left() > 0 and d_fails < self.search_cfg.trials_depth:
            base_snap = self.model.snapshot()
            if self.model.num_layers + 1 > self.search_cfg.max_layers:
                break
            self.model.append_layer()
            prop_val = self._train_budgeted(self.model, train_loader, val_loader)
            if self._accept(base_val, prop_val):
                base_val = prop_val
                d_fails = 0
                # run width mini-series
                w_fails = 0
                while self._budget_left() > 0 and w_fails < self.search_cfg.trials_width:
                    series_snap = self.model.snapshot()
                    if self.model.hidden_size + self.search_cfg.ex_k > self.search_cfg.max_hidden:
                        break
                    self.model.widen_hidden(self.search_cfg.ex_k)
                    prop2_val = self._train_budgeted(self.model, train_loader, val_loader)
                    if self._accept(base_val, prop2_val):
                        base_val = prop2_val
                        w_fails = 0
                    else:
                        self.model = TextRNN.from_snapshot(series_snap)
                        w_fails += 1
            else:
                self.model = TextRNN.from_snapshot(base_snap)
                d_fails += 1
        return self.model

    def search_alt_width_first(self, train_loader, val_loader):
        base_val = self._train_budgeted(self.model, train_loader, val_loader)
        while self._budget_left() > 0:
            improved = False
            # width step
            base_snap = self.model.snapshot()
            if self.model.hidden_size + self.search_cfg.ex_k <= self.search_cfg.max_hidden:
                self.model.widen_hidden(self.search_cfg.ex_k)
                prop = self._train_budgeted(self.model, train_loader, val_loader)
                if self._accept(base_val, prop):
                    base_val = prop
                    improved = True
                else:
                    self.model = TextRNN.from_snapshot(base_snap)
            # depth mini-series
            d_fails = 0
            while self._budget_left() > 0 and d_fails < self.search_cfg.trials_depth:
                series_snap = self.model.snapshot()
                if self.model.num_layers + 1 > self.search_cfg.max_layers:
                    break
                self.model.append_layer()
                prop2 = self._train_budgeted(self.model, train_loader, val_loader)
                if self._accept(base_val, prop2):
                    base_val = prop2
                    improved = True
                    d_fails = 0
                else:
                    self.model = TextRNN.from_snapshot(series_snap)
                    d_fails += 1
            if not improved:
                break
        return self.model

    def search_alt_depth_first(self, train_loader, val_loader):
        base_val = self._train_budgeted(self.model, train_loader, val_loader)
        while self._budget_left() > 0:
            improved = False
            # depth step
            base_snap = self.model.snapshot()
            if self.model.num_layers + 1 <= self.search_cfg.max_layers:
                self.model.append_layer()
                prop = self._train_budgeted(self.model, train_loader, val_loader)
                if self._accept(base_val, prop):
                    base_val = prop
                    improved = True
                else:
                    self.model = TextRNN.from_snapshot(base_snap)
            # width mini-series
            w_fails = 0
            while self._budget_left() > 0 and w_fails < self.search_cfg.trials_width:
                series_snap = self.model.snapshot()
                if self.model.hidden_size + self.search_cfg.ex_k > self.search_cfg.max_hidden:
                    break
                self.model.widen_hidden(self.search_cfg.ex_k)
                prop2 = self._train_budgeted(self.model, train_loader, val_loader)
                if self._accept(base_val, prop2):
                    base_val = prop2
                    improved = True
                    w_fails = 0
                else:
                    self.model = TextRNN.from_snapshot(series_snap)
                    w_fails += 1
            if not improved:
                break
        return self.model


# ==============================
# Simple text dataset (tokenized elsewhere)
# ==============================

class TensorTextDataset(Dataset):
    """A minimal dataset of (token_ids, length, label). lengths optional."""
    def __init__(self, X: torch.Tensor, y: torch.Tensor, lengths: Optional[torch.Tensor]=None):
        assert X.dim() == 2, "X must be (N, T)"
        assert y.dim() == 1 and y.size(0) == X.size(0), "y must be (N,) and match X"
        self.X = X.long()
        self.y = y.long()
        self.lengths = lengths.long() if lengths is not None else None

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        if self.lengths is None:
            return self.X[idx], self.y[idx]
        return {"x": self.X[idx], "y": self.y[idx], "lengths": self.lengths[idx]}
