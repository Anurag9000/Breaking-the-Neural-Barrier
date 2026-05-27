import copy
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore
from utils.adp_logging import ContinuousLogger
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Load baseline
BASELINE_PATH = Path(__file__).with_name("mlp_ssl_stl.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASELINE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)
MLPSSL = baseline_module.MLPSSL  # type: ignore
MLPEncoder = baseline_module.MLPEncoder  # type: ignore
ProjectionHead = baseline_module.ProjectionHead  # type: ignore


# ADP REVIEW (BEFORE REFACTOR)
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Inner training: train_with_patience ties ES reset to delta and reloads immediately.
# ADP REVIEW: delegated to utils.adp_contract forward-only core.
# - Control flow: toggles modes on no improvement; lacks forward-only march and context-end restore per updated spec.
# - ES patience conflated with expansion patiences; no snapshot/restore separation.


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 5
    trials_width: int = 10
    trials_depth: int = 5
    ex_k: int = 1
    width_stage_margin_patience: int = 5
    width_stage_min_improve_pct: float = 1.0
    max_width: int = 4096
    max_depth: int = 10
    width_stage_margin_patience: int = 5
    width_stage_min_improve_pct: float = 1.0
    depth_stage_margin_patience: int = 5
    depth_stage_min_improve_pct: float = 1.0
    min_new_layer_width: int = 10
    depth_first_seed_width: int = 20
    max_neurons: int = 10_000_000
    min_new_layer_width: int = 10
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 128
    val_split: float = 0.1
    max_epochs: int = 100_000_000
    temperature: float = 0.2


def _resize_linear(old: nn.Linear, new_out: int, new_in: int) -> nn.Linear:
    new = nn.Linear(new_in, new_out, bias=old.bias is not None).to(old.weight.device)
    with torch.no_grad():
        r = min(old.out_features, new_out)
        c = min(old.in_features, new_in)
        new.weight[:r, :c] = old.weight[:r, :c]
        if old.bias is not None and new.bias is not None:
            new.bias[:r] = old.bias[:r]
    return new


def rebuild_encoder(enc: MLPEncoder, hidden_widths: List[int]):
    device = next(enc.parameters()).device
    in_dim = enc.in_dim
    rep_dim = enc.rep_dim
    use_bn = enc.use_bn
    layers = []
    prev = in_dim
    old_layers = list(enc.backbone)
    for w in hidden_widths:
        block = baseline_module.MLPBlock(prev, w, use_bn).to(device)  # type: ignore
        if old_layers:
            old_block = old_layers.pop(0)
            block.linear = _resize_linear(old_block.linear, w, prev)
        layers.append(block)
        prev = w
    enc.backbone = nn.Sequential(*layers)
    enc.rep = _resize_linear(enc.rep, rep_dim, prev)
    enc.hidden_widths = hidden_widths


def rebuild_projector(ph: ProjectionHead, in_dim: int):
    ph.fc1 = _resize_linear(ph.fc1, in_dim, in_dim)
    ph.fc2 = _resize_linear(ph.fc2, ph.fc2.out_features, in_dim)


def _next_staged_widths(hidden_widths: List[int], max_width: int, ex_k: int) -> List[int]:
    widths = [int(w) for w in hidden_widths]
    if not widths:
        return widths
    target = min(max(widths) + max(1, int(ex_k)), int(max_width)) if len(set(widths)) == 1 else max(widths)
    next_widths = list(widths)
    for idx, width in enumerate(next_widths):
        if width < target:
            next_widths[idx] = width + 1
            break
    return next_widths


def expand_width(model: MLPSSL, ex_k: int, max_width: int) -> Optional[MLPSSL]:
    new_h = _next_staged_widths(model.encoder.hidden_widths, max_width, ex_k)
    if new_h == model.encoder.hidden_widths:
        return None
    rebuild_encoder(model.encoder, new_h)
    rebuild_projector(model.projector, new_h[-1]) # Projector input depends on encoder output
    return model


def expand_depth(model: MLPSSL, max_depth: int) -> Optional[MLPSSL]:
    if len(model.encoder.hidden_widths) >= max_depth:
        return None
    if len(set(int(w) for w in model.encoder.hidden_widths)) != 1:
        return None
    new_h = model.encoder.hidden_widths + [model.encoder.hidden_widths[-1]]
    rebuild_encoder(model.encoder, new_h)
    rebuild_projector(model.projector, new_h[-1])
    return model


def total_neurons(model: MLPSSL) -> int:
    return sum(model.encoder.hidden_widths)


def snapshot_arch_and_state(model: MLPSSL, state_dict=None) -> Dict[str, Any]:
    state = state_dict if state_dict is not None else model.state_dict()
    return {
        "in_dim": model.encoder.in_dim,
        "rep_dim": model.encoder.rep_dim,
        "proj_dim": model.projector.fc2.out_features,
        "hidden_widths": list(model.encoder.hidden_widths),
        "use_bn": model.encoder.use_bn,
        "state": copy.deepcopy(state)
    }


def restore_arch_and_state(model: MLPSSL, snap: Dict[str, Any], device) -> MLPSSL:
    # Rebuild
    new_model = MLPSSL(
        in_dim=snap["in_dim"],
        hidden_widths=snap["hidden_widths"],
        rep_dim=snap["rep_dim"],
        proj_dim=snap["proj_dim"],
        use_bn=snap["use_bn"]
    ).to(device)
    new_model.load_state_dict(snap["state"])
    return new_model


def make_loaders(batch_size: int, val_split: float):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    return dl_train, dl_val


def nt_xent_loss(p_i, p_j, temperature: float = 0.2):
    z_i = F.normalize(p_i, dim=1)
    z_j = F.normalize(p_j, dim=1)
    N = z_i.size(0)
    z = torch.cat([z_i, z_j], dim=0)
    sim = torch.matmul(z, z.T) / temperature
    sim = sim - torch.eye(2 * N, device=sim.device) * 1e9
    targets = torch.cat([torch.arange(N, 2 * N), torch.arange(0, N)], dim=0).to(sim.device)
    loss = F.cross_entropy(sim, targets)
    return loss


def train_with_early_stopping(model: MLPSSL, dl_train, dl_val, acfg: ADPConfig, device) -> Tuple[float, Dict[str, Any]]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best_val = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    es_counter = 0
    
    for _ in range(acfg.max_epochs):
        model.train()
        for (x1, _), (x2, _) in zip(dl_train, dl_train):
            x1 = x1.to(device)
            x2 = x2.to(device)
            opt.zero_grad(set_to_none=True)
            _, p1 = model(x1)
            _, p2 = model(x2)
            loss = nt_xent_loss(p1, p2, acfg.temperature)
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for x, _ in dl_val:
                x = x.to(device)
                _, p = model(x)
                val_loss += (p.pow(2).sum(dim=1).mean()).item()
        val_loss /= max(len(dl_val), 1)
        
        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            es_counter = 0
        else:
            es_counter += 1
            
        if es_counter >= acfg.patience:
            break
            
    return best_val, best_state


def adp_search(model: MLPSSL, dl_train, dl_val, acfg: ADPConfig, device):
    
    # Initial training
    from utils.adp_contract import run_module_adp
    from utils.adp_introspect import infer_adp_shape

    best_val, model = run_module_adp(
        globals(),
        model,
        dl_train,
        dl_val,
        acfg,
        device,
        log_loss=locals().get("log_loss", False),
        log_neurons=locals().get("log_neurons", False),
        results_dir=locals().get("results_dir"),
        logger=locals().get("logger"),
    )

    return best_val, model


def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP MLP SSL (SimCLR-style) width/depth search")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--rep-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["alt_width", "alt_depth", "width_to_depth", "depth_to_width"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--trials-width", type=int, default=10)
    p.add_argument("--trials-depth", type=int, default=5)
    p.add_argument("--ex-k", type=int, default=1)
    p.add_argument("--width-stage-margin-patience", type=int, default=5)
    p.add_argument("--width-stage-min-improve-pct", type=float, default=1.0)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--max-epochs", type=int, default=100000000)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1)
    in_dim = 3 * 32 * 32
    model = MLPSSL(in_dim, hidden_widths=args.hidden, rep_dim=args.rep_dim, proj_dim=args.proj_dim)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    acfg = ADPConfig(
        adp_mode=args.adp_mode,
        delta=args.delta,
        patience=args.patience,
        trials_width=args.trials_width,
        trials_depth=args.trials_depth,
        ex_k=args.ex_k,
        width_stage_margin_patience=args.width_stage_margin_patience,
        width_stage_min_improve_pct=args.width_stage_min_improve_pct,
        max_width=args.max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )
    best, model = adp_search(model.to(device), dl_train, dl_val, acfg, device)
    print(f"[ADP MLP SSL] mode={args.adp_mode} best_val={best:.6f} hidden={model.encoder.hidden_widths} depth={len(model.encoder.hidden_widths)+1}")
