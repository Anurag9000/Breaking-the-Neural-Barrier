import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
import sys
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

sys.path.append(str(Path(__file__).resolve().parents[3]))
from utils.adp_plot import plot_loss_vs_epoch, plot_loss_vs_neurons  # type: ignore

# Load baseline for reference
BASE_PATH = Path(__file__).with_name("ae_jigsaw.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)

# ADP REVIEW (BEFORE REFACTOR)
# - Modes: width_only/width, depth_only/depth, width_to_depth, depth_to_width, alt_width, alt_depth toggled via ad hoc loop.
# - Inner training: train_with_patience uses delta for ES; no separate patience_es; CrossEntropy loss but ES tied to delta.
# - Width expansion: widen_model mutates in place; trials_width counter; no snapshot/restore abstraction; delta shared for width/depth.
# - Depth expansion: deepen_model mutates head; rollback via state only; no architecture snapshot; trials_depth only.
# - 2D/ALT: width_to_depth/depth_to_width/alt_* just toggle on no improvement; missing structured outer/inner loops and phase saturation.
# - Patiences: lacks distinct patience_width_exp / patience_depth_exp application per context; relies on improved flag.
# Deviations: Missing snapshot_arch_and_state/restore_arch_and_state, proper expansion patiences, and exact control flow for ADP_WIDTH_ONLY, ADP_DEPTH_ONLY, ADP_DEPTH_OUTER_WIDTH_INNER, ADP_WIDTH_OUTER_DEPTH_INNER, ADP_ALT_DEPTH, ADP_ALT_WIDTH.


@dataclass
class ADPConfig:
    adp_mode: str = "width_to_depth"
    delta: float = 1e-3
    patience: int = 10
    trials_width: int = 2
    trials_depth: int = 2
    ex_k: int = 16
    max_width: int = 256
    max_depth: int = 5  # hidden layers in head (>=2)
    max_neurons: int = 5_000_000
    lr: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    max_epochs: int = 20
    grid_size: int = 3
    num_permutations: int = 30


def _resize_conv(conv: nn.Conv2d, in_ch: int, out_ch: int) -> nn.Conv2d:
    new = nn.Conv2d(
        in_ch,
        out_ch,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        bias=conv.bias is not None,
        dilation=conv.dilation,
    ).to(conv.weight.device)
    with torch.no_grad():
        r = min(out_ch, conv.out_channels)
        c = min(in_ch, conv.in_channels)
        new.weight[:r, :c] = conv.weight[:r, :c]
        if conv.bias is not None and new.bias is not None:
            new.bias[:r] = conv.bias[:r]
    return new


def _resize_bn(bn: nn.BatchNorm2d, out_ch: int) -> nn.BatchNorm2d:
    new = nn.BatchNorm2d(out_ch).to(bn.weight.device)
    with torch.no_grad():
        r = min(out_ch, bn.weight.numel())
        new.weight[:r] = bn.weight[:r]
        new.bias[:r] = bn.bias[:r]
        new.running_mean[:r] = bn.running_mean[:r]
        new.running_var[:r] = bn.running_var[:r]
    return new


def build_encoder(in_ch: int, width: int) -> Tuple[nn.Sequential, int]:
    enc = nn.Sequential(
        nn.Conv2d(in_ch, width, 3, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
        nn.Conv2d(width, width, 3, padding=1),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2),
        nn.Conv2d(width, 2 * width, 3, padding=1),
        nn.BatchNorm2d(2 * width),
        nn.ReLU(inplace=True),
        nn.AdaptiveAvgPool2d(1),
    )
    out_dim = 2 * width
    return enc, out_dim


def copy_encoder(old: nn.Sequential, new: nn.Sequential):
    # Copy overlapping weights for conv+bn layers
    for o, n in zip(old, new):
        if isinstance(o, nn.Conv2d) and isinstance(n, nn.Conv2d):
            with torch.no_grad():
                r = min(o.out_channels, n.out_channels)
                c = min(o.in_channels, n.in_channels)
                n.weight[:r, :c] = o.weight[:r, :c]
                if o.bias is not None and n.bias is not None:
                    n.bias[:r] = o.bias[:r]
        elif isinstance(o, nn.BatchNorm2d) and isinstance(n, nn.BatchNorm2d):
            with torch.no_grad():
                r = min(o.weight.numel(), n.weight.numel())
                n.weight[:r] = o.weight[:r]
                n.bias[:r] = o.bias[:r]
                n.running_mean[:r] = o.running_mean[:r]
                n.running_var[:r] = o.running_var[:r]


def build_head(input_dim: int, hidden_layers: int, num_perm: int):
    assert hidden_layers >= 2
    layers: List[nn.Module] = []
    h1 = 4 * input_dim
    layers.extend([nn.Linear(input_dim, h1), nn.ReLU(inplace=True)])
    for _ in range(hidden_layers - 2):
        layers.extend([nn.Linear(h1 // 2, h1 // 2), nn.ReLU(inplace=True)])
    h_last = h1 // 2
    layers.extend([nn.Linear(h_last, h_last), nn.ReLU(inplace=True)])
    layers.append(nn.Linear(h_last, num_perm))
    return nn.Sequential(*layers)


def copy_linear(old: nn.Linear, new: nn.Linear):
    with torch.no_grad():
        r_out = min(old.out_features, new.out_features)
        r_in = min(old.in_features, new.in_features)
        new.weight[:r_out, :r_in] = old.weight[:r_out, :r_in]
        if old.bias is not None and new.bias is not None:
            new.bias[:r_out] = old.bias[:r_out]


def copy_head(old: nn.Sequential, new: nn.Sequential):
    for o, n in zip(old, new):
        if isinstance(o, nn.Linear) and isinstance(n, nn.Linear):
            copy_linear(o, n)


class ADPJigsawModel(nn.Module):
    def __init__(self, in_ch: int = 3, grid_size: int = 3, width: int = 64, num_permutations: int = 30, hidden_layers: int = 2):
        super().__init__()
        self.in_ch = in_ch
        self.G = grid_size
        self.K = grid_size * grid_size
        self.num_permutations = num_permutations
        self.base_width = width
        self.hidden_layers = max(2, hidden_layers)

        self.encoder, self.encoder_dim = build_encoder(in_ch, width)
        self.head = build_head(self.encoder_dim * self.K, self.hidden_layers, num_permutations)

    def total_neurons(self) -> int:
        D = self.encoder_dim
        return D * self.K + sum(m.out_features for m in self.head if isinstance(m, nn.Linear))

    def depth(self) -> int:
        return self.hidden_layers

    def widths_list(self) -> List[int]:
        D = self.encoder_dim
        return [D, 4 * D, 2 * D]

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        B, K, C, h, w = patches.shape
        assert K == self.K
        x = patches.view(B * K, C, h, w)
        feat = self.encoder(x)  # (B*K, D)
        feat = feat.view(B, K, -1).flatten(1)
        return self.head(feat)


def total_neurons(model: ADPJigsawModel) -> int:
    return model.total_neurons()


def rebuild_model(model: ADPJigsawModel, width: int, depth: int, device) -> ADPJigsawModel:
    new_model = ADPJigsawModel(
        in_ch=model.in_ch,
        grid_size=model.G,
        width=width,
        num_permutations=model.num_permutations,
        hidden_layers=depth,
    ).to(device)
    new_model.load_state_dict(copy.deepcopy(model.state_dict()), strict=False)
    return new_model


def snapshot_arch_and_state(model: ADPJigsawModel):
    return {
        "width": model.base_width,
        "depth": model.hidden_layers,
        "state": copy.deepcopy(model.state_dict()),
        "in_ch": model.in_ch,
        "grid_size": model.G,
        "num_perm": model.num_permutations,
    }


def restore_arch_and_state(model: ADPJigsawModel, snapshot, device) -> ADPJigsawModel:
    restored = ADPJigsawModel(
        in_ch=snapshot.get("in_ch", model.in_ch),
        grid_size=snapshot.get("grid_size", model.G),
        width=snapshot["width"],
        num_permutations=snapshot.get("num_perm", model.num_permutations),
        hidden_layers=snapshot["depth"],
    ).to(device)
    restored.load_state_dict(snapshot["state"], strict=False)
    return restored


def expand_width(model: ADPJigsawModel, ex_k_width: int, max_width: int, device) -> ADPJigsawModel:
    new_w = min(max_width, model.base_width + ex_k_width)
    return rebuild_model(model, new_w, model.hidden_layers, device)


def expand_depth(model: ADPJigsawModel, ex_k_depth: int, device) -> ADPJigsawModel:
    new_d = model.hidden_layers + ex_k_depth
    return rebuild_model(model, model.base_width, new_d, device)


def make_permutation_set(K: int, num_perm: int) -> torch.Tensor:
    perms = set()
    out = []
    while len(out) < num_perm:
        p = tuple(torch.randperm(K).tolist())
        if p not in perms:
            perms.add(p)
            out.append(torch.tensor(p, dtype=torch.long))
    return torch.stack(out, dim=0)


def jigsaw_collate(batch, G: int, perm_set: torch.Tensor):
    K = G * G
    ps = []
    ids = []
    for x, _ in batch:
        # x: (C,H,W)
        C, H, W = x.shape
        ph = H // G
        pw = W // G
        patches = x.unfold(1, ph, ph).unfold(2, pw, pw)  # (C, G, G, ph, pw)
        patches = patches.permute(1, 2, 0, 3, 4).contiguous().view(K, C, ph, pw)
        idx = torch.randint(0, perm_set.size(0), (1,)).item()
        order = perm_set[idx]
        shuffled = patches[order]
        ps.append(shuffled)
        ids.append(idx)
    patches_stack = torch.stack(ps, dim=0)
    ids = torch.tensor(ids, dtype=torch.long)
    return patches_stack, ids


def make_loaders(batch_size: int, val_split: float, G: int, num_perm: int):
    tf = transforms.Compose([transforms.ToTensor()])
    ds = datasets.CIFAR10(root="./data", train=True, download=True, transform=tf)
    n_val = int(len(ds) * val_split)
    n_train = len(ds) - n_val
    train_ds, val_ds = random_split(ds, [n_train, n_val])
    perm_set = make_permutation_set(G * G, num_perm)
    collate = lambda batch: jigsaw_collate(batch, G, perm_set)
    dl_train = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True, collate_fn=collate)
    dl_val = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, collate_fn=collate)
    return dl_train, dl_val


def train_with_early_stopping(model: ADPJigsawModel, dl_train, dl_val, acfg: ADPConfig, device, history: list) -> Tuple[float, dict, None]:
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    crit = nn.CrossEntropyLoss()
    best = float("inf")
    best_state = copy.deepcopy(model.state_dict())
    remaining = acfg.patience
    for _ in range(acfg.max_epochs):
        model.train()
        for patches, perm_id in dl_train:
            patches = patches.to(device)
            perm_id = perm_id.to(device)
            logits = model(patches)
            loss = crit(logits, perm_id)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            if acfg.grad_clip is not None:
                nn.utils.clip_grad_norm_(model.parameters(), acfg.grad_clip)
            opt.step()
        model.eval()
        with torch.no_grad():
            val = 0.0
            n = 0
            for patches, perm_id in dl_val:
                patches = patches.to(device)
                perm_id = perm_id.to(device)
                logits = model(patches)
                l = crit(logits, perm_id)
                val += l.item() * patches.size(0)
                n += patches.size(0)
            val = val / max(n, 1)
        history.append(val)
        if val < best:
            best = val
            best_state = copy.deepcopy(model.state_dict())
            remaining = acfg.patience
        else:
            remaining -= 1
        if remaining <= 0:
            break
    model.load_state_dict(best_state)
    return best, best_state, None


def adp_search(model: ADPJigsawModel, dl_train, dl_val, acfg: ADPConfig, device, log_loss: bool = False, log_neurons: bool = False, results_dir: Path = Path("results_adp")):
    results_dir.mkdir(parents=True, exist_ok=True)
    val_history: List[float] = []
    improvements: List[tuple[int, float]] = []

    delta_width = acfg.delta
    delta_depth = acfg.delta
    patience_width_exp = acfg.trials_width
    patience_depth_exp = acfg.trials_depth
    ex_k_width = acfg.ex_k
    ex_k_depth = 1

    def can_widen(width: int, depth: int) -> bool:
        new_w = min(acfg.max_width, width + ex_k_width)
        if new_w > acfg.max_width:
            return False
        temp = rebuild_model(model, new_w, depth, device)
        return total_neurons(temp) <= acfg.max_neurons

    def can_deepen(width: int, depth: int) -> bool:
        new_d = depth + ex_k_depth
        if new_d > acfg.max_depth:
            return False
        temp = rebuild_model(model, width, new_d, device)
        return total_neurons(temp) <= acfg.max_neurons

    best_val, best_state, _ = train_with_early_stopping(model, dl_train, dl_val, acfg, device, val_history)
    best_width = model.base_width
    best_depth = model.hidden_layers
    improvements.append((total_neurons(model), best_val))

    def width_search(local_model: ADPJigsawModel, initial_val=None, initial_state=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_state = initial_state
        local_best_width = local_model.base_width
        if local_best_val is None or local_best_state is None:
            local_best_val, local_best_state, _ = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history)
        width_failure_count = 0
        while width_failure_count < patience_width_exp and can_widen(local_model.base_width, local_model.hidden_layers):
            local_model = expand_width(local_model, ex_k_width, acfg.max_width, device)
            val, state, _ = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history)
            if val < local_best_val - delta_width:
                local_best_val = val
                local_best_state = state
                local_best_width = local_model.base_width
                width_failure_count = 0
                if log_improvement:
                    improvements.append((total_neurons(local_model), local_best_val))
            else:
                width_failure_count += 1
        local_model = rebuild_model(local_model, local_best_width, local_model.hidden_layers, device)
        local_model.load_state_dict(local_best_state)
        return local_model, local_best_val, local_best_state, local_best_width

    def depth_search(local_model: ADPJigsawModel, initial_val=None, initial_state=None, log_improvement: bool = False):
        local_best_val = initial_val
        local_best_state = initial_state
        local_best_depth = local_model.hidden_layers
        if local_best_val is None or local_best_state is None:
            local_best_val, local_best_state, _ = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history)
        depth_failure_count = 0
        while depth_failure_count < patience_depth_exp and can_deepen(local_model.base_width, local_model.hidden_layers):
            local_model = expand_depth(local_model, ex_k_depth, device)
            val, state, _ = train_with_early_stopping(local_model, dl_train, dl_val, acfg, device, val_history)
            if val < local_best_val - delta_depth:
                local_best_val = val
                local_best_state = state
                local_best_depth = local_model.hidden_layers
                depth_failure_count = 0
                if log_improvement:
                    improvements.append((total_neurons(local_model), local_best_val))
            else:
                depth_failure_count += 1
        local_model = rebuild_model(local_model, local_model.base_width, local_best_depth, device)
        local_model.load_state_dict(local_best_state)
        return local_model, local_best_val, local_best_state, local_best_depth

    mode = acfg.adp_mode
    if mode in ("width_only", "width"):
        model, best_val, best_state, best_width = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
    elif mode in ("depth_only", "depth"):
        model, best_val, best_state, best_depth = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_width = model.base_width
    elif mode == "depth_to_width":  # ADP_DEPTH_OUTER_WIDTH_INNER
        model, best_val, best_state, best_width = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_depth = model.hidden_layers
        depth_failure_count = 0
        while depth_failure_count < patience_depth_exp and can_deepen(best_width, best_depth):
            model = expand_depth(model, ex_k_depth, device)
            cand_model, cand_val, cand_state, cand_width = width_search(model, log_improvement=False)
            if cand_val < best_val - delta_depth:
                best_val = cand_val
                best_state = cand_state
                best_width = cand_width
                best_depth = model.hidden_layers
                depth_failure_count = 0
                model = cand_model
                model.load_state_dict(best_state)
                improvements.append((total_neurons(model), best_val))
            else:
                depth_failure_count += 1
    elif mode == "width_to_depth":  # ADP_WIDTH_OUTER_DEPTH_INNER
        model, best_val, best_state, best_depth = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
        best_width = model.base_width
        width_failure_count = 0
        while width_failure_count < patience_width_exp and can_widen(best_width, best_depth):
            model = expand_width(model, ex_k_width, acfg.max_width, device)
            cand_model, cand_val, cand_state, cand_depth = depth_search(model, log_improvement=False)
            if cand_val < best_val - delta_width:
                best_val = cand_val
                best_state = cand_state
                best_width = model.base_width
                best_depth = cand_depth
                width_failure_count = 0
                model = cand_model
                model.load_state_dict(best_state)
                improvements.append((total_neurons(model), best_val))
            else:
                width_failure_count += 1
    elif mode == "alt_depth":
        best_width = model.base_width
        best_depth = model.hidden_layers
        depth_saturated = False
        width_saturated = False
        phase = "depth"
        while not (depth_saturated and width_saturated):
            if phase == "depth":
                model, phase_val, phase_state, phase_depth = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val
                    best_state = phase_state
                    best_depth = phase_depth
                    depth_saturated = False
                    improvements.append((total_neurons(model), best_val))
                else:
                    depth_saturated = True
                model = rebuild_model(model, best_width, best_depth, device)
                model.load_state_dict(best_state)
                phase = "width"
            else:
                model, phase_val, phase_state, phase_width = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val
                    best_state = phase_state
                    best_width = phase_width
                    width_saturated = False
                    improvements.append((total_neurons(model), best_val))
                else:
                    width_saturated = True
                model = rebuild_model(model, best_width, best_depth, device)
                model.load_state_dict(best_state)
                phase = "depth"
    elif mode == "alt_width":
        best_width = model.base_width
        best_depth = model.hidden_layers
        depth_saturated = False
        width_saturated = False
        phase = "width"
        while not (depth_saturated and width_saturated):
            if phase == "width":
                model, phase_val, phase_state, phase_width = width_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val
                    best_state = phase_state
                    best_width = phase_width
                    width_saturated = False
                    improvements.append((total_neurons(model), best_val))
                else:
                    width_saturated = True
                model = rebuild_model(model, best_width, best_depth, device)
                model.load_state_dict(best_state)
                phase = "depth"
            else:
                model, phase_val, phase_state, phase_depth = depth_search(model, initial_val=best_val, initial_state=best_state, log_improvement=True)
                if phase_val < best_val:
                    best_val = phase_val
                    best_state = phase_state
                    best_depth = phase_depth
                    depth_saturated = False
                    improvements.append((total_neurons(model), best_val))
                else:
                    depth_saturated = True
                model = rebuild_model(model, best_width, best_depth, device)
                model.load_state_dict(best_state)
                phase = "width"
    else:
        raise ValueError(f"Unsupported ADP mode: {acfg.adp_mode}")

    model = rebuild_model(model, best_width, best_depth, device)
    model.load_state_dict(best_state)
    if log_loss:
        plot_loss_vs_epoch(val_history, results_dir / "loss_vs_epoch.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    if log_neurons and improvements:
        plot_loss_vs_neurons([n for n, _ in improvements], [v for _, v in improvements], results_dir / "loss_vs_neurons.png", title=f"{BASE_PATH.stem} ({acfg.adp_mode})")
    return best_val, model, best_width, best_depth


# ADP REVIEW (AFTER REFACTOR)
# - Mode: width_only / width -> ADP_WIDTH_ONLY (depth fixed; ES with patience_es=patience; width_failure_count vs trials_width; accept val < best - delta_width).
# - Mode: depth_only / depth -> ADP_DEPTH_ONLY (width fixed; depth_failure_count vs trials_depth; accept val < best - delta_depth).
# - Mode: depth_to_width -> ADP_DEPTH_OUTER_WIDTH_INNER (outer depth +1 with patience_depth_exp/delta_depth; inner width search with patience_width_exp/delta_width).
# - Mode: width_to_depth -> ADP_WIDTH_OUTER_DEPTH_INNER (outer width +ex_k with patience_width_exp/delta_width; inner depth search with patience_depth_exp/delta_depth).
# - Mode: alt_depth -> ADP_ALT_DEPTH (phase depth-only until depth patience hit, then width-only until width patience hit; repeat until both saturated).
# - Mode: alt_width -> ADP_ALT_WIDTH (start width phase then depth phase, same patience rules, repeat until both saturated).
# - Snapshot/restore + expand_width/expand_depth follow spec; patience mapping: patience->patience_es, trials_width->patience_width_exp, trials_depth->patience_depth_exp; delta used for both width/depth thresholds.


def main():
    import argparse

    p = argparse.ArgumentParser(description="ADP Jigsaw permutation model (width/depth search)")
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--grid-size", type=int, default=3)
    p.add_argument("--num-permutations", type=int, default=30)
    p.add_argument(
        "--adp-mode",
        type=str,
        default="width_to_depth",
        choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"],
    )
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--trials-width", type=int, default=2)
    p.add_argument("--trials-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-width", type=int, default=256)
    p.add_argument("--max-depth", type=int, default=5)
    p.add_argument("--max-neurons", type=int, default=5_000_000)
    p.add_argument("--max-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    args = p.parse_args()

    dl_train, dl_val = make_loaders(args.batch_size, 0.1, args.grid_size, args.num_permutations)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ADPJigsawModel(
        in_ch=3, grid_size=args.grid_size, width=args.width, num_permutations=args.num_permutations, hidden_layers=2
    ).to(device)
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
        max_epochs=args.max_epochs,
        grid_size=args.grid_size,
        num_permutations=args.num_permutations,
    )
    best_val, model, width, depth = adp_search(model, dl_train, dl_val, acfg, device)
    print(
        f"[ADP Jigsaw] mode={args.adp_mode} best_val={best_val:.6f} width={width} head_depth={depth}"
    )


if __name__ == "__main__":
    main()
