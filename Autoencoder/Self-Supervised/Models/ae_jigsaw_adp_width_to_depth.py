import copy
from dataclasses import dataclass
from pathlib import Path
import importlib.util
from typing import List, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# Load baseline for reference
BASE_PATH = Path(__file__).with_name("ae_jigsaw.py").resolve()
_spec = importlib.util.spec_from_file_location("baseline_module", BASE_PATH)
baseline_module = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(baseline_module)


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


def widen_model(model: ADPJigsawModel, ex_k: int, max_width: int):
    new_w = min(max_width, model.base_width + ex_k)
    if new_w == model.base_width:
        return
    new_encoder, new_dim = build_encoder(model.in_ch, new_w)
    copy_encoder(model.encoder, new_encoder)
    model.encoder = new_encoder
    model.base_width = new_w
    model.encoder_dim = new_dim
    old_head = model.head
    model.head = build_head(model.encoder_dim * model.K, model.hidden_layers, model.num_permutations)
    copy_head(old_head, model.head)


def deepen_model(model: ADPJigsawModel):
    model.hidden_layers += 1
    old_head = model.head
    model.head = build_head(model.encoder_dim * model.K, model.hidden_layers, model.num_permutations)
    copy_head(old_head, model.head)


def total_neurons(model: ADPJigsawModel) -> int:
    return model.total_neurons()


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


def train_with_patience(model: ADPJigsawModel, dl_train, dl_val, acfg: ADPConfig, device):
    opt = torch.optim.AdamW(model.parameters(), lr=acfg.lr, weight_decay=acfg.weight_decay)
    crit = nn.CrossEntropyLoss()
    best = float("inf")
    best_state = None
    pat = acfg.patience
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
        if val < best - acfg.delta:
            best = val
            best_state = copy.deepcopy(model.state_dict())
            pat = acfg.patience
        else:
            pat -= 1
        if pat <= 0:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return best


def adp_search(model: ADPJigsawModel, dl_train, dl_val, acfg: ADPConfig, device):
    def can_widen():
        next_w = model.base_width + acfg.ex_k
        return next_w <= acfg.max_width and (next_w * 2 * model.K) < acfg.max_neurons

    def can_deepen():
        return (model.hidden_layers + 1) <= acfg.max_depth and total_neurons(model) < acfg.max_neurons

    inner_val = train_with_patience(model, dl_train, dl_val, acfg, device)
    best_val, best_state = inner_val, copy.deepcopy(model.state_dict())
    pw, pd = acfg.trials_width, acfg.trials_depth
    mode = acfg.adp_mode
    improved = True
    while improved:
        improved = False
        if mode in ("width_only", "width", "width_to_depth", "alt_width"):
            if can_widen() and pw > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_w = model.base_width
                pre_val = inner_val
                widen_model(model, acfg.ex_k, acfg.max_width)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pw = acfg.trials_width
                    improved = True
                    if v < best_val:
                        best_val, best_state = v, copy.deepcopy(model.state_dict())
                else:
                    model.base_width = pre_w
                    model.load_state_dict(pre)
                    pw -= 1
            if mode == "width_only":
                continue
        if mode in ("depth_only", "depth", "depth_to_width", "alt_depth"):
            if can_deepen() and pd > 0:
                pre = copy.deepcopy(model.state_dict())
                pre_depth = model.hidden_layers
                pre_val = inner_val
                deepen_model(model)
                v = train_with_patience(model, dl_train, dl_val, acfg, device)
                if v < pre_val - acfg.delta:
                    inner_val = v
                    pd = acfg.trials_depth
                    improved = True
                    if v < best_val:
                        best_val, best_state = v, copy.deepcopy(model.state_dict())
                else:
                    model.hidden_layers = pre_depth
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
    best = adp_search(model, dl_train, dl_val, acfg, device)
    print(
        f"[ADP Jigsaw] mode={args.adp_mode} best_val={best:.6f} width={model.base_width} head_depth={model.hidden_layers}"
    )


if __name__ == "__main__":
    main()
