
import argparse, torch
from torchvision import datasets, transforms

from adp_vae_sup_model import (ConvVAE_Sup, TrainConfig, SearchConfig,
                               recon_loss, kl_loss,
                               build_vae,
                               width_to_depth, depth_to_width,
                               alt_depth_first, alt_width_first,
                               depth_only, width_only)

# -----------------------------
# Data loaders (supervised)
# -----------------------------
def make_loaders(dataset: str, data_root: str, batch_size: int, num_workers: int = 2):
    ds = dataset.lower()
    if ds == "mnist":
        in_ch, img_size, num_classes = 1, 28, 10
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.MNIST(root=data_root, train=True, transform=tfm, download=True)
        test  = datasets.MNIST(root=data_root, train=False, transform=tfm, download=True)
    elif ds == "fashionmnist":
        in_ch, img_size, num_classes = 1, 28, 10
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.FashionMNIST(root=data_root, train=True, transform=tfm, download=True)
        test  = datasets.FashionMNIST(root=data_root, train=False, transform=tfm, download=True)
    elif ds == "cifar10":
        in_ch, img_size, num_classes = 3, 32, 10
        tfm = transforms.Compose([transforms.ToTensor()])
        train = datasets.CIFAR10(root=data_root, train=True, transform=tfm, download=True)
        test  = datasets.CIFAR10(root=data_root, train=False, transform=tfm, download=True)
    else:
        raise ValueError(f"Unknown dataset {dataset}. Choose mnist|fashionmnist|cifar10")
    dl_train = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=num_workers, drop_last=True)
    dl_val   = torch.utils.data.DataLoader(test,  batch_size=batch_size, shuffle=False, num_workers=num_workers, drop_last=False)
    return dl_train, dl_val, in_ch, img_size, num_classes

def main():
    p = argparse.ArgumentParser()
    # Data
    p.add_argument("--dataset", type=str, default="cifar10", choices=["mnist","fashionmnist","cifar10"])
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--num-workers", type=int, default=2)
    # Model
    p.add_argument("--init-width", type=int, default=64)
    p.add_argument("--init-depth", type=int, default=4)
    p.add_argument("--z-dim", type=int, default=64)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--down-every", type=int, default=2)
    p.add_argument("--recon", type=str, default="bce", choices=["bce","mse","l1"])
    # Supervision
    p.add_argument("--sup-mode", type=str, default="aux", choices=["aux","cvae"])
    p.add_argument("--aux-head", action="store_true", help="enable aux classifier when in CVAE mode")
    p.add_argument("--lambda-sup", type=float, default=1.0)
    p.add_argument("--min-acc-drop", type=float, default=0.0)
    # Training
    p.add_argument("--max-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--es-patience", type=int, default=10)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--beta", type=float, default=4.0)
    p.add_argument("--kl-warmup-epochs", type=int, default=20)
    p.add_argument("--noise-std", type=float, default=0.1)
    p.add_argument("--cutout-p", type=float, default=0.0)
    # ADP
    p.add_argument("--adp-strategy", type=str, default="alt_width",
                   choices=["width_to_depth","depth_to_width","alt_depth","alt_width","depth_only","width_only"])
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience-width", type=int, default=2)
    p.add_argument("--patience-depth", type=int, default=2)
    p.add_argument("--ex-k", type=int, default=16)
    p.add_argument("--max-neurons", type=int, default=3000000)
    p.add_argument("--max-depth", type=int, default=16)
    p.add_argument("--max-width", type=int, default=1024)
    p.add_argument("--max-total-epochs", type=int, default=None)

    args = p.parse_args()

    dl_train, dl_val, in_ch, img_size, num_classes = make_loaders(args.dataset, args.data_root, args.batch_size, args.num_workers)
    net = build_vae(in_ch, img_size, args.init_width, args.init_depth, args.z_dim, num_classes, args.dropout,
                    down_every=args.down_every, recon=args.recon, sup_mode=args.sup_mode, aux_head=(args.sup_mode=="aux" or args.aux_head))

    tcfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, es_patience=args.es_patience,
                       grad_clip=args.grad_clip, beta=args.beta, recon=args.recon,
                       kl_warmup_epochs=args.kl_warmup_epochs, noise_std=args.noise_std, cutout_p=args.cutout_p,
                       lambda_sup=args.lambda_sup, sup_mode=args.sup_mode, aux_head=(args.sup_mode=="aux" or args.aux_head),
                       min_acc_drop=args.min_acc_drop)
    scfg = SearchConfig(delta=args.delta, patience_width=args.patience_width, patience_depth=args.patience_depth,
                        ex_k=args.ex_k, max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
                        max_total_epochs=args.max_total_epochs, down_every=args.down_every)

    # pick strategy
    if args.adp_strategy == "width_to_depth":
        net = width_to_depth(net, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    elif args.adp_strategy == "depth_to_width":
        net = depth_to_width(net, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    elif args.adp_strategy == "alt_depth":
        net = alt_depth_first(net, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    elif args.adp_strategy == "alt_width":
        net = alt_width_first(net, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    elif args.adp_strategy == "depth_only":
        net = depth_only(net, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    elif args.adp_strategy == "width_only":
        net = width_only(net, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    else:
        raise ValueError("Unknown strategy")

    # final validation readout
    net.eval(); net.to(tcfg.device)
    with torch.no_grad():
        tot, n = 0.0, 0; correct = 0
        for x, y in dl_val:
            x = x.to(tcfg.device); y = y.to(tcfg.device)
            x_logits, mu, logvar, logits = net(x, y if args.sup_mode=="cvae" else None)
            loss = recon_loss(x_logits, x, args.recon) + tcfg.beta * kl_loss(mu, logvar)
            tot += float(loss.item()) * x.size(0); n += x.size(0)
            if getattr(net, "aux_head_on", False) and logits is not None:
                pred = logits.argmax(dim=1); correct += int((pred == y).sum().item())
        acc = (correct / n) if getattr(net, "aux_head_on", False) else 0.0
        print(f"[VAL] elbo_like={tot/max(n,1):.6f}  acc={acc:.4f}  neurons={net.total_neurons()}  depth={len(net.channels)}  widths={net.widths}  z_dim={net.z_dim}  mode={args.sup_mode}  strategy={args.adp_strategy}")

if __name__ == "__main__":
    main()
