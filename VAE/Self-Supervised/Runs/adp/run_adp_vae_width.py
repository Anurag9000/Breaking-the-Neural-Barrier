
import argparse, torch
from adp_vae_core import (ConvVAE, TrainConfig, SearchConfig, make_loaders, build_vae,
                          vae_width_to_depth, vae_depth_to_width,
                          vae_alt_depth_first, vae_alt_width_first,
                          vae_depth_only, vae_width_only)

def build(parser):
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["mnist","fashionmnist","cifar10"])
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=2)

    parser.add_argument("--init-width", type=int, default=64)
    parser.add_argument("--init-depth", type=int, default=4)
    parser.add_argument("--z-dim", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--down-every", type=int, default=2)
    parser.add_argument("--recon", type=str, default="bce", choices=["bce","mse","l1"])

    parser.add_argument("--max-epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--es-patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--beta", type=float, default=4.0)
    parser.add_argument("--kl-warmup-epochs", type=int, default=20)
    parser.add_argument("--noise-std", type=float, default=0.1)
    parser.add_argument("--cutout-p", type=float, default=0.0)
    parser.add_argument("--lambda-align", type=float, default=0.0)
    parser.add_argument("--lambda-mmd", type=float, default=0.0)

    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--patience-width", type=int, default=2)
    parser.add_argument("--patience-depth", type=int, default=2)
    parser.add_argument("--ex-k", type=int, default=16)
    parser.add_argument("--max-neurons", type=int, default=3000000)
    parser.add_argument("--max-depth", type=int, default=16)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--max-total-epochs", type=int, default=None)
    return parser

def get_common(args):
    dl_train, dl_val, in_ch, img_size = make_loaders(args.dataset, args.data_root, args.batch_size, args.num_workers)
    net = build_vae(in_ch, img_size, args.init_width, args.init_depth, args.z_dim, args.dropout,
                    down_every=args.down_every, recon=args.recon)
    tcfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, es_patience=args.es_patience,
                       grad_clip=args.grad_clip, beta=args.beta, recon=args.recon,
                       kl_warmup_epochs=args.kl_warmup_epochs, noise_std=args.noise_std, cutout_p=args.cutout_p,
                       lambda_align=args.lambda_align, lambda_mmd=args.lambda_mmd)
    scfg = SearchConfig(delta=args.delta, patience_width=args.patience_width, patience_depth=args.patience_depth,
                        ex_k=args.ex_k, max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
                        max_total_epochs=args.max_total_epochs, down_every=args.down_every)
    return net, tcfg, scfg, dl_train, dl_val

def final_eval(net, dl_val, device):
    import torch
    net.eval(); net.to(device)
    tot, n = 0.0, 0
    from adp_vae_core import recon_loss, kl_loss, add_noise, cutout
    with torch.no_grad():
        e = 999  # end of training
        for x, _ in dl_val:
            x = x.to(device)
            x_noisy = add_noise(x, 0.0)
            x_noisy = cutout(x_noisy, p=0.0)
            x_logits, mu, logvar = net(x_noisy)
            loss = recon_loss(x_logits, x, "bce") + 1.0 * kl_loss(mu, logvar)
            tot += float(loss.item()) * x.size(0); n += x.size(0)
    print(f"[VAL] elbo_like={tot/max(n,1):.6f}  neurons={net.total_neurons()}  depth={len(net.channels)}  widths={net.widths}  z_dim={net.z_dim}")

def main():
    parser = build(argparse.ArgumentParser())
    args = parser.parse_args()
    net, tcfg, scfg, dl_train, dl_val = get_common(args)
    from adp_vae_core import vae_width_to_depth
    net = vae_width_to_depth(net, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    final_eval(net, dl_val, tcfg.device)
if __name__ == "__main__": main()
