import copy
import importlib.util
from dataclasses import dataclass
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
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


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 20
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 128
    max_width: int = 4096
    max_depth: int = 10
    max_neurons: int = 10_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    batch_size: int = 128
    val_split: float = 0.1
    max_epochs: int = 50
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


def widen_all(model: MLPSSL, ex_k: int, max_width: int):
    new_h = [min(max_width, w + ex_k) for w in model.encoder.hidden_widths]
    rebuild_encoder(model.encoder, new_h)
    rebuild_projector(model.projector, new_h[-1])


def append_depth(model: MLPSSL):
    new_h = model.encoder.hidden_widths + [model.encoder.hidden_widths[-1]]
    rebuild_encoder(model.encoder, new_h)
    rebuild_projector(model.projector, new_h[-1])


def total_neurons(model: MLPSSL) -> int:
    return sum(model.encoder.hidden_widths)


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


def train_with_patience(model: MLPSSL, dl_train, dl_val, acfg: ADPConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    best = float("inf")
    best_state = None
    pat = acfg.patience
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
        if val_loss < best - 1e-9:
            best = val_loss
            best_state = copy.deepcopy(model.state_dict())
            pat = acfg.patience
        else:
            pat -= 1
        if pat <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def adp_search(model: MLPSSL, dl_train, dl_val, acfg: ADPConfig, device):
    def can_widen():
        return max(model.encoder.hidden_widths) + acfg.ex_k <= acfg.max_width and total_neurons(model) < acfg.max_neurons

    def can_deepen():
        return len(model.encoder.hidden_widths) + 1 <= acfg.max_depth and (total_neurons(model) + model.encoder.hidden_widths[-1]) <= acfg.max_neurons

    inner_val = train_with_patience(model, dl_train, dl_val, acfg, device)
    best_val = inner_val
    best_state = copy.deepcopy(model.state_dict())

    pw = acfg.trials_width
    pd = acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only", "width", "width_to_depth", "alt_width"):
            if can_widen() and pw > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_val = inner_val
                widen_all(model, acfg.ex_k, acfg.max_width)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pw = acfg.trials_width
                    improved = True
                    if v < best_val:
                        best_val = v
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre)
                    pw -= 1
            if mode == "width_only":
                continue
        if mode in ("depth_only", "depth", "depth_to_width", "alt_depth"):
            if can_deepen() and pd > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_val = inner_val
                append_depth(model)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pd = acfg.trials_depth
                    improved = True
                    if v < best_val:
                        best_val = v
                        best_state = copy.deepcopy(model.state_dict())
                else:
                    model.load_state_dict(pre)
                    pd -= 1
            if mode == "depth_only":
                continue
        if mode == "width_to_depth" and not improved:
            mode = "depth"
            pd = acfg.trials_depth
            improved = True
        elif mode == "depth_to_width" and not improved:
            mode = "width"
            pw = acfg.trials_width
            improved = True
        elif mode in ("alt_width", "alt_depth"):
            mode = "depth" if mode == "alt_width" else "width"
            improved = True

    model.load_state_dict(best_state)
    return best_val


def main():
    import argparse
    p = argparse.ArgumentParser(description="ADP MLP SSL (SimCLR-style) width/depth search")
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--rep-dim", type=int, default=256)
    p.add_argument("--proj-dim", type=int, default=128)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=128)
    p.add_argument("--max-width", type=int, default=4096)
    p.add_argument("--max-depth", type=int, default=10)
    p.add_argument("--max-neurons", type=int, default=10_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
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
        max_width=args.max_width,
        max_depth=args.max_depth,
        max_neurons=args.max_neurons,
        batch_size=args.batch_size,
        max_epochs=args.max_epochs,
    )
    best = adp_search(model.to(device), dl_train, dl_val, acfg, device)
    print(f\"[ADP MLP SSL] mode={args.adp_mode} best_val={best:.6f} hidden={model.encoder.hidden_widths} depth={len(model.encoder.hidden_widths)+1}\")


if __name__ == "__main__":
    main()
