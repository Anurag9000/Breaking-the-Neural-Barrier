
import argparse, os, random, torch
from torch.utils.data import DataLoader
from adp_mlp_ssl_depth import AdaptiveMLPSSL, build_data, adp_search_depth_then_width

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist","fashionmnist","cifar10","cifar100"])
    p.add_argument("--data_dir", type=str, default="./data")
    p.add_argument("--img_size", type=int, nargs=2, default=[32,32])
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--hidden", type=int, nargs="+", default=[1024, 1024])
    p.add_argument("--rep_dim", type=int, default=256)
    p.add_argument("--proj_dim", type=int, default=128)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--ex_k", type=int, default=128)
    p.add_argument("--trials_depth", type=int, default=3)
    p.add_argument("--trials_width", type=int, default=3)
    p.add_argument("--max_neurons", type=int, default=16384)
    p.add_argument("--max_depth", type=int, default=30)
    p.add_argument("--max_width", type=int, default=4096)
    p.add_argument("--val_split", type=float, default=0.1)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tr, va, shape = build_data(args.dataset, args.data_dir, args.img_size, args.val_split)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True, drop_last=True)

    in_dim = shape[0]*shape[1]*shape[2]
    model = AdaptiveMLPSSL(in_dim, args.hidden, args.rep_dim, args.proj_dim, use_bn=True)

    best_val = adp_search_depth_then_width(model, trl, val, device,
                                           trials_depth=args.trials_depth, trials_width=args.trials_width,
                                           epochs=args.epochs, lr=args.lr, patience=args.patience,
                                           delta=args.delta, ex_k=args.ex_k,
                                           max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
                                           temperature=args.temperature)

    os.makedirs("checkpoints", exist_ok=True)
    path = os.path.join("checkpoints", f"adp_mlp_ssl_depth_{args.dataset}.pt")
    torch.save({"state": {k: v.cpu() for k, v in model.state_dict().items()}, "best_val_ntxent": best_val, "config": vars(args)}, path)
    print(f"Saved best model to {path} (val_ntxent={best_val:.6f})")

if __name__ == "__main__":
    main()
