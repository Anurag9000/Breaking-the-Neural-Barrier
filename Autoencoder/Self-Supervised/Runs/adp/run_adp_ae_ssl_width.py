import argparse, torch
from adp_ae_ssl_core import (AutoencoderSSL, TrainConfig, SearchConfig, make_cifar10_ssl_loaders,
                             ae_ssl_width_to_depth, ae_ssl_depth_to_width,
                             ae_ssl_alt_depth_first, ae_ssl_alt_width_first,
                             ae_ssl_depth_only, ae_ssl_width_only)

def build(parser):
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--two-views", action="store_true")
    parser.add_argument("--init-width", type=int, default=16)
    parser.add_argument("--init-depth", type=int, default=2)
    parser.add_argument("--proj-dim", type=int, default=None)
    parser.add_argument("--pool-idx", type=int, nargs="*", default=[0,2])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--es-patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-consistency", type=float, default=1.0)
    parser.add_argument("--lambda-barlow", type=float, default=0.0)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--patience-width", type=int, default=2)
    parser.add_argument("--patience-depth", type=int, default=2)
    parser.add_argument("--ex-k", type=int, default=8)
    parser.add_argument("--max-neurons", type=int, default=400000)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--max-total-epochs", type=int, default=None)
    return parser

def get_common(args):
    dl_train, dl_val, dl_test = make_cifar10_ssl_loaders(
        data_root=args.data_root, batch_size=args.batch_size, num_workers=args.num_workers,
        val_split=args.val_split, download=args.download, two_views=args.two_views
    )
    widths = [args.init_width] * args.init_depth
    model = AutoencoderSSL(in_ch=3, widths=widths, pooling_indices=args.pool_idx, proj_dim=args.proj_dim)
    tcfg = TrainConfig(
        lr=args.lr, weight_decay=args.weight_decay, es_patience=args.es_patience, grad_clip=args.grad_clip,
        lambda_recon=args.lambda_recon, lambda_consistency=args.lambda_consistency, lambda_barlow=args.lambda_barlow,
        projector_dim=args.proj_dim, two_views=args.two_views
    )
    scfg = SearchConfig(
        delta=args.delta, patience_width=args.patience_width, patience_depth=args.patience_depth,
        ex_k=args.ex_k, max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
        max_total_epochs=args.max_total_epochs, pooling_indices=tuple(args.pool_idx)
    )
    return model, tcfg, scfg, dl_train, dl_val, dl_test

def final_eval(model, dl_test, device):
    model.eval(); model.to(device)
    import torch.nn.functional as F
    tot, n = 0.0, 0
    with torch.no_grad():
        for x, _ in dl_test:
            x = x.to(device); rec, _ = model(x)
            tot += F.mse_loss(rec, x, reduction="sum").item(); n += x.size(0)
    print(f"[TEST] recon_mse={tot/n:.4f}  neurons={model.total_neurons()}  depth={len(model.encoder)}  widths={model.widths}")

def main():
    parser = build(argparse.ArgumentParser())
    args = parser.parse_args()
    model, tcfg, scfg, dl_train, dl_val, dl_test = get_common(args)
    model = ae_ssl_width_to_depth(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    final_eval(model, dl_test, tcfg.device)
if __name__ == "__main__": main()
