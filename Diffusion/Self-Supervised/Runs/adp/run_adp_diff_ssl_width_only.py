import argparse, torch
from adp_diff_ssl_core import (AdaptiveUNet, TrainConfig, SearchConfig, DiffConfig,
                               make_cifar10_loaders_diff,
                               diff_width_to_depth, diff_depth_to_width,
                               diff_alt_depth_first, diff_alt_width_first,
                               diff_depth_only, diff_width_only)

def build(parser):
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--init-width", type=int, default=64)
    parser.add_argument("--init-depth", type:int, default=3)
    parser.add_argument("--pool-idx", type=int, nargs="*", default=[0,2])
    parser.add_argument("--max-epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--es-patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--patience-width", type=int, default=2)
    parser.add_argument("--patience-depth", type=int, default=2)
    parser.add_argument("--ex-k", type=int, default=16)
    parser.add_argument("--max-neurons", type=int, default=1200000)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--max-width", type:int, default=1024)
    parser.add_argument("--max-total-epochs", type=int, default=None)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--beta-start", type=float, default=1e-4)
    parser.add_argument("--beta-end", type=float, default=2e-2)
    return parser

def get_common(args):
    dl_train, dl_val, dl_test = make_cifar10_loaders_diff(
        data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
        val_split=args.val_split, download=args.download
    )
    widths = [args.init_width] * args.init_depth
    net = AdaptiveUNet(in_ch=3, widths=widths, pooling_indices=args.pool_idx, emb_dim=256)
    tcfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, es_patience=args.es_patience, grad_clip=args.grad_clip)
    scfg = SearchConfig(delta=args.delta, patience_width=args.patience_width, patience_depth=args.patience_depth,
                        ex_k=args.ex_k, max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
                        max_total_epochs=args.max_total_epochs, pooling_indices=tuple(args.pool_idx))
    dcfg = DiffConfig(timesteps=args.timesteps, beta_start=args.beta_start, beta_end=args.beta_end)
    return net, tcfg, scfg, dcfg, dl_train, dl_val, dl_test

def final_eval(net, dl_test, device):
    # Report validation-style DDPM noise MSE on test set (lower is better).
    import torch
    import torch.nn.functional as F
    from adp_diff_ssl_core import DiffusionHelper, DiffConfig
    net.eval(); net.to(device)
    diff = DiffusionHelper(DiffConfig(), device)
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, _ in dl_test:
            x = x.to(device)
            B = x.size(0)
            t = torch.randint(0, diff.T, (B,), device=device, dtype=torch.long)
            x_noisy, eps = diff.q_sample(x, t)
            eps_pred = net(x_noisy, t)
            loss = F.mse_loss(eps_pred, eps, reduction="sum")
            tot += float(loss.item()); n += B
    print(f"[TEST] noise_mse={tot/n:.6f}  neurons={net.total_neurons()}  depth={len(net.encoder)}  widths={net.widths}")

def main():
    parser = build(argparse.ArgumentParser())
    args = parser.parse_args()
    net, tcfg, scfg, dcfg, dl_train, dl_val, dl_test = get_common(args)
    net = diff_width_only(net, dl_train, dl_val, tcfg, scfg, dcfg, max_epochs=args.max_epochs)
    final_eval(net, dl_test, tcfg.device)
if __name__ == "__main__": main()
