
import argparse, os, random, torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from adp_mlp_ae_width_only import AdaptiveMLPAE, adp_search_width_only

def build_data(dataset, data_dir, img_size, val_split):
    tfm = transforms.Compose([transforms.Resize(img_size), transforms.ToTensor()])
    dataset = dataset.lower()
    if dataset == "mnist":
        ds = datasets.MNIST(data_dir, train=True, download=True, transform=tfm)
        C=1
    elif dataset == "fashionmnist":
        ds = datasets.FashionMNIST(data_dir, train=True, download=True, transform=tfm)
        C=1
    elif dataset == "cifar10":
        ds = datasets.CIFAR10(data_dir, train=True, download=True, transform=tfm)
        C=3
    elif dataset == "cifar100":
        ds = datasets.CIFAR100(data_dir, train=True, download=True, transform=tfm)
        C=3
    else:
        raise ValueError("Unsupported dataset")
    val_len = int(len(ds) * val_split)
    tr_len = len(ds) - val_len
    tr, va = random_split(ds, [tr_len, val_len])
    return tr, va, (C, img_size[0], img_size[1])

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist","fashionmnist","cifar10","cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--img_size", type=int, nargs=2, default=[32,32])
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 512])
    p.add_argument("--bottleneck", type=int, default=256)
    p.add_argument("--delta", type=float, default=1e-4)
    p.add_argument("--ex_k", type=int, default=128)
    p.add_argument("--trials_width", type=int, default=5)
    p.add_argument("--max_neurons", type=int, default=8192)
    p.add_argument("--max_depth", type=int, default=25)
    p.add_argument("--max_width", type=int, default=4096)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--denoise_std", type=float, default=0.0, help=">0.0 enables denoising AE")
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, shape = build_data(args.dataset, args.data_dir, args.img_size, args.val_split)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    in_dim = shape[0]*shape[1]*shape[2]
    model = AdaptiveMLPAE(in_dim, args.hidden, args.bottleneck, use_bn=True, output_activation="sigmoid")

    best_val = adp_search_width_only(model, trl, val, device,
                                     trials_width=args.trials_width,
                                     epochs=args.epochs, lr=args.lr, patience=args.patience,
                                     delta=args.delta, ex_k=args.ex_k,
                                     max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
                                     denoise_std=args.denoise_std)

    os.makedirs("checkpoints", exist_ok=True)
    ckpt = {
        "state": {k: v.cpu() for k, v in model.state_dict().items()},
        "best_val_mse": best_val,
        "config": vars(args)
    }
    path = os.path.join("checkpoints", f"adp_mlp_ae_width_only_{args.dataset}.pt")
    torch.save(ckpt, path)
    print(f"Saved best model to {path} (val_mse={best_val:.6f})")

if __name__ == "__main__":
    main()
