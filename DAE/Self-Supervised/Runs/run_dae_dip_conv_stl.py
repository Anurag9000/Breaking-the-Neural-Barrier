import argparse
import csv
import json
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision as tv
import torchvision.transforms as T

from ..Models.dae_dip_conv_stl import DAEDIPConv, dip_total_neurons


def build_single_image_loader(
    dataset: str,
    data_dir: str,
    batch_size: int,
) -> DataLoader:
    if dataset.lower() == "cifar10":
        mean, std = (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)
        ds_class = tv.datasets.CIFAR10
    elif dataset.lower() == "cifar100":
        mean, std = (0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)
        ds_class = tv.datasets.CIFAR100
    else:
        raise ValueError("dataset must be cifar10 or cifar100")

    tf = T.Compose(
        [
            T.ToTensor(),
            T.Normalize(mean, std),
        ]
    )
    ds = ds_class(root=data_dir, train=True, transform=tf, download=True)
    # Deep image prior is per-image; we just use the first image here.
    x0, _ = ds[0]
    x0 = x0.unsqueeze(0)  # (1,C,H,W)
    tensor_ds = torch.utils.data.TensorDataset(x0)
    return DataLoader(tensor_ds, batch_size=batch_size, shuffle=False)


def main() -> None:
    p = argparse.ArgumentParser(description="Deep-image-prior Conv DAE STL on single CIFAR image")
    p.add_argument("--dataset", type=str, default="cifar10", choices=["cifar10", "cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--out_dir", type=str, default="./runs/dae_dip_conv_stl")
    p.add_argument("--seed", type=int, default=1337)
    # Model
    p.add_argument("--width", type=int, default=64)
    p.add_argument("--depth", type=int, default=5)
    # Training
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=1e-3)

    args = p.parse_args()

    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = build_single_image_loader(args.dataset, args.data_dir, batch_size=1)
    x0 = next(iter(loader))[0].to(device)  # (1,C,H,W)

    model = DAEDIPConv(in_channels=3, width=args.width, depth=args.depth).to(device)
    opt = optim.Adam(model.parameters(), lr=args.lr)
    mse = nn.MSELoss()

    # Fixed input noise
    z = torch.randn_like(x0)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log_path = out_dir / "training_log.txt"
    stats_path = out_dir / "training_stats.csv"
    img_path = out_dir / "dip_reconstruction.pt"

    log_f = log_path.open("w", encoding="utf-8")
    stats_f = stats_path.open("w", newline="", encoding="utf-8")
    stats_writer = csv.writer(stats_f)
    stats_writer.writerow(["epoch", "width", "depth", "neurons", "train_loss"])

    neurons = dip_total_neurons(args.width, args.depth)
    best_loss = float("inf")
    best_epoch = -1
    best_rec = None

    try:
        for epoch in range(1, args.epochs + 1):
            model.train()
            opt.zero_grad(set_to_none=True)
            x_rec, _ = model(z)
            loss = mse(x_rec, x0)
            loss.backward()
            opt.step()

            l = float(loss.item())
            if l < best_loss:
                best_loss = l
                best_epoch = epoch
                best_rec = x_rec.detach().cpu()

            msg = f"Epoch {epoch:05d} | loss={l:.6f} | best={best_loss:.6f} @ {best_epoch}"
            if epoch % 50 == 0 or epoch == 1:
                print(msg)
            log_f.write(msg + "\n")
            stats_writer.writerow([epoch, args.width, args.depth, neurons, l])
            stats_f.flush()
    finally:
        log_f.flush()
        stats_f.flush()

    if best_rec is not None:
        torch.save({"reconstruction": best_rec, "target": x0.cpu()}, img_path)

    report = {
        "dataset": args.dataset,
        "width": args.width,
        "depth": args.depth,
        "neurons_metric": neurons,
        "best_loss": best_loss,
        "best_epoch": best_epoch,
    }
    with (out_dir / "report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    log_f.write("\n" + json.dumps(report, indent=2) + "\n")
    log_f.close()
    stats_f.close()


if __name__ == "__main__":
    main()

