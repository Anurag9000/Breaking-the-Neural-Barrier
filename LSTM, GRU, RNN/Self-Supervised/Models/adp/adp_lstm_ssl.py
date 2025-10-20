
import math
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


def _make_lstm(input_size: int, hidden_size: int, num_layers: int, bidirectional: bool=False, dropout: float=0.0) -> nn.LSTM:
    return nn.LSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bias=True,
        batch_first=True,
        dropout=dropout if num_layers > 1 else 0.0,
        bidirectional=bidirectional
    )


def _resize_lstm_hidden(old: nn.LSTM, new_hidden: int) -> nn.LSTM:
    # Return a new LSTM with resized hidden_size, overlap-copying the weights/biases.
    new = _make_lstm(
        input_size=old.input_size,
        hidden_size=new_hidden,
        num_layers=old.num_layers,
        bidirectional=old.bidirectional,
        dropout=old.dropout,
    )
    # Copy all layers directions
    num_dirs = 2 if old.bidirectional else 1
    with torch.no_grad():
        for layer in range(old.num_layers):
            for d in range(num_dirs):
                suffix = f"_l{layer}"
                if d == 1:
                    suffix += "_reverse"
                # weight_ih
                _copy_param_slice_(getattr(new, f"weight_ih{suffix}"), getattr(old, f"weight_ih{suffix}"))
                # weight_hh
                _copy_param_slice_(getattr(new, f"weight_hh{suffix}"), getattr(old, f"weight_hh{suffix}"))
                # bias_ih / bias_hh
                _copy_param_slice_(getattr(new, f"bias_ih{suffix}"), getattr(old, f"bias_ih{suffix}"))
                _copy_param_slice_(getattr(new, f"bias_hh{suffix}"), getattr(old, f"bias_hh{suffix}"))
    return new


def _append_lstm_layer(old: nn.LSTM) -> nn.LSTM:
    # Return a new LSTM with num_layers+1, overlap-copying existing layers and
    # initializing the last layer from the previous top layer (when shapes permit).
    new = _make_lstm(
        input_size=old.input_size,
        hidden_size=old.hidden_size,
        num_layers=old.num_layers + 1,
        bidirectional=old.bidirectional,
        dropout=old.dropout,
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
        # init last layer from previous top layer (best-effort copy)
        prev = old.num_layers - 1
        for d in range(num_dirs):
            src_suffix = f"_l{prev}"
            dst_suffix = f"_l{old.num_layers}"  # new last layer index
            if d == 1:
                src_suffix += "_reverse"
                dst_suffix += "_reverse"
            for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
                _copy_param_slice_(getattr(new, f"{name}{dst_suffix}"), getattr(old, f"{name}{src_suffix}"))
    return new


# ==============================
# Self-supervised LSTM LM
# ==============================

class LSTMLanguageModel(nn.Module):
    """Causal LM: predict next token for each time step. Single-model, no teacher."""
    def __init__(
        self,
        vocab_size: int,
        emb_dim: int = 256,
        hidden_size: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
        pad_idx: int = 0,
        bidirectional: bool = False,  # keep False for causal LM
        tie_weights: bool = True,
    ):
        super().__init__()
        assert not bidirectional, "Causal LM must be unidirectional."
        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout_p = dropout
        self.pad_idx = pad_idx
        self.tie_weights = tie_weights

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx)
        self.lstm = _make_lstm(emb_dim, hidden_size, num_layers, bidirectional=False, dropout=dropout)
        self.dropout = nn.Dropout(dropout)
        self.decoder = nn.Linear(hidden_size, vocab_size, bias=False)

        if tie_weights:
            # Tie decoder weight with embedding for parameter sharing
            if emb_dim != hidden_size:
                # bridge to equal dims
                self.proj = nn.Linear(hidden_size, emb_dim, bias=False)
                self.decoder.weight = self.embedding.weight  # decoder after proj uses tied weights implicitly at head
            else:
                self.proj = nn.Identity()
                self.decoder.weight = self.embedding.weight
        else:
            self.proj = nn.Identity()

    # ----- growth ops (ADP) -----
    def widen_hidden(self, delta: int):
        self.hidden_size += delta
        self.lstm = _resize_lstm_hidden(self.lstm, self.hidden_size)
        # update projection/decoder shapes
        if isinstance(self.proj, nn.Identity) and self.emb_dim != self.hidden_size:
            self.proj = nn.Linear(self.hidden_size, self.emb_dim, bias=False)
        elif isinstance(self.proj, nn.Linear):
            self.proj = nn.Linear(self.hidden_size, self.proj.out_features, bias=False)  # reinit then overlap-copy not necessary

        self.decoder = nn.Linear(self.emb_dim if self.tie_weights else self.hidden_size, self.vocab_size, bias=False)
        if self.tie_weights:
            self.decoder.weight = self.embedding.weight

    def append_layer(self):
        self.num_layers += 1
        self.lstm = _append_lstm_layer(self.lstm)

    # ----- snapshot/restore -----
    def snapshot(self) -> Dict:
        return {
            "state": self.state_dict(),
            "arch": {
                "vocab_size": self.vocab_size,
                "emb_dim": self.emb_dim,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout_p,
                "pad_idx": self.pad_idx,
                "tie_weights": self.tie_weights,
            }
        }

    @staticmethod
    def from_snapshot(snap: Dict) -> "LSTMLanguageModel":
        arch = snap["arch"]
        model = LSTMLanguageModel(**arch)
        model.load_state_dict(snap["state"])
        return model

    def forward(self, tokens, lengths=None):
        # tokens: (B, T)
        emb = self.embedding(tokens)  # (B, T, E)
        out, _ = self.lstm(emb)       # (B, T, H)
        out = self.dropout(out)
        out = self.proj(out)          # (B, T, E) if tied, else identity
        logits = self.decoder(out)    # (B, T, V)
        return logits


# ==============================
# Configs
# ==============================

@dataclass
class TrainCfg:
    lr: float = 3e-4
    weight_decay: float = 0.0
    batch_size: int = 64
    max_epochs: int = 200
    es_patience: int = 10
    clip_grad: float = 1.0


@dataclass
class SearchCfg:
    delta: float = 0.0
    ex_k: int = 64            # hidden-size widen step
    trials_width: int = 50
    trials_depth: int = 50
    max_total_epochs: int = 300
    max_layers: int = 12
    max_hidden: int = 2048


# ==============================
# Train / Eval for LM
# ==============================

def lm_loss_and_acc(logits: torch.Tensor, targets: torch.Tensor, pad_idx: int=0) -> Tuple[torch.Tensor, float]:
    # logits: (B, T, V), targets: (B, T)
    B, T, V = logits.shape
    logits = logits[:, :-1, :].contiguous().view(-1, V)  # predict next token for all but last
    targets = targets[:, 1:].contiguous().view(-1)       # next tokens
    loss = nn.functional.cross_entropy(logits, targets, ignore_index=pad_idx, reduction='mean')
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = targets != pad_idx
        correct = (pred[mask] == targets[mask]).sum().item()
        total = mask.sum().item()
        acc = correct / max(total, 1)
    return loss, acc


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device, pad_idx: int=0) -> Tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_tok = 0
    total_correct = 0
    with torch.no_grad():
        for batch in loader:
            x = batch if isinstance(batch, torch.Tensor) else batch[0]
            x = x.to(device)
            logits = model(x)
            loss, acc = lm_loss_and_acc(logits, x, pad_idx=pad_idx)
            B, T = x.shape
            total_loss += loss.item() * B
            total_correct += acc * B
            total_tok += B
    avg_loss = total_loss / max(total_tok, 1)
    avg_acc = total_correct / max(total_tok, 1)
    return avg_loss, avg_acc


def train_until_plateau(model: nn.Module, train_loader: DataLoader, val_loader: DataLoader, device: torch.device, cfg: TrainCfg, max_epochs: int, pad_idx: int=0) -> Tuple[float, Dict, int]:
    model = model.to(device)
    opt = optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_snap = model.state_dict()
    best_val = float("inf")
    epochs_run = 0
    es_counter = 0
    for epoch in range(min(cfg.max_epochs, max_epochs)):
        model.train()
        for batch in train_loader:
            x = batch if isinstance(batch, torch.Tensor) else batch[0]
            x = x.to(device)
            logits = model(x)
            loss, _ = lm_loss_and_acc(logits, x, pad_idx=pad_idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.clip_grad is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=cfg.clip_grad)
            opt.step()

        val_loss, _ = evaluate(model, val_loader, device, pad_idx=pad_idx)
        epochs_run += 1
        if val_loss + 1e-9 < best_val:
            best_val = val_loss
            best_snap = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            es_counter = 0
        else:
            es_counter += 1
        if es_counter >= cfg.es_patience:
            break
    model.load_state_dict(best_snap)
    return best_val, best_snap, epochs_run


# ==============================
# ADP search (six strategies)
# ==============================

class ADP_LSTM_SSL:
    def __init__(self, model: LSTMLanguageModel, train_cfg: TrainCfg, search_cfg: SearchCfg, device: Optional[torch.device]=None, pad_idx: int=0):
        self.model = model
        self.train_cfg = train_cfg
        self.search_cfg = search_cfg
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.global_epochs = 0
        self.pad_idx = pad_idx

    def _accept(self, base_loss: float, prop_loss: float) -> bool:
        return prop_loss < (base_loss - self.search_cfg.delta)

    def _budget_left(self) -> int:
        return max(0, self.search_cfg.max_total_epochs - self.global_epochs)

    def _train_budgeted(self, model: LSTMLanguageModel, train_loader, val_loader) -> float:
        remaining = self._budget_left()
        if remaining <= 0:
            return float("inf")
        val, snap, spent = train_until_plateau(model, train_loader, val_loader, self.device, self.train_cfg, remaining, pad_idx=self.pad_idx)
        self.global_epochs += spent
        return val

    # ---- strategies ----

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
                self.model = LSTMLanguageModel.from_snapshot(base_snap)
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
                self.model = LSTMLanguageModel.from_snapshot(base_snap)
                failures += 1
        return self.model

    def search_width_to_depth(self, train_loader, val_loader):
        base_val = self._train_budgeted(self.model, train_loader, val_loader)
        w_fails = 0
        while self._budget_left() > 0 and w_fails < self.search_cfg.trials_width:
            base_snap = self.model.snapshot()
            if self.model.hidden_size + self.search_cfg.ex_k > self.search_cfg.max_hidden:
                break
            self.model.widen_hidden(self.search_cfg.ex_k)
            prop_val = self._train_budgeted(self.model, train_loader, val_loader)
            if self._accept(base_val, prop_val):
                base_val = prop_val
                w_fails = 0
                # mini depth series
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
                        self.model = LSTMLanguageModel.from_snapshot(series_snap)
                        d_fails += 1
            else:
                self.model = LSTMLanguageModel.from_snapshot(base_snap)
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
                # mini width series
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
                        self.model = LSTMLanguageModel.from_snapshot(series_snap)
                        w_fails += 1
            else:
                self.model = LSTMLanguageModel.from_snapshot(base_snap)
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
                    self.model = LSTMLanguageModel.from_snapshot(base_snap)
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
                    self.model = LSTMLanguageModel.from_snapshot(series_snap)
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
                    self.model = LSTMLanguageModel.from_snapshot(base_snap)
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
                    self.model = LSTMLanguageModel.from_snapshot(series_snap)
                    w_fails += 1
            if not improved:
                break
        return self.model


# ==============================
# Simple token dataset for LM
# ==============================

class TensorLM(Dataset):
    """Minimal dataset of token sequences (B, T)."""
    def __init__(self, X: torch.Tensor):
        assert X.dim() == 2, "X must be (N, T)"
        self.X = X.long()

    def __len__(self):
        return self.X.size(0)

    def __getitem__(self, idx):
        return self.X[idx]
