import argparse, os, torch
from adp_ae_core import (AutoencoderCNN, TrainConfig, SearchConfig, make_cifar10_loaders,
                         ae_search_width_to_depth, ae_search_depth_to_width,
                         ae_search_alt_depth_first, ae_search_alt_width_first,
                         ae_search_depth_only, ae_search_width_only)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--init-width", type=int, default=16)
    parser.add_argument("--init-depth", type=int, default=2)
    parser.add_argument("--pool-idx", type=int, nargs="*", default=[0,2], help="0-based indices for MaxPool after encoder blocks")
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--es-patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--lambda-recon", type=float, default=1.0)
    parser.add_argument("--lambda-cls", type=float, default=1.0)
    parser.add_argument("--delta", type=float, default=1e-3)
    parser.add_argument("--patience-width", type=int, default=2)
    parser.add_argument("--patience-depth", type=int, default=2)
    parser.add_argument("--ex-k", type=int, default=8)
    parser.add_argument("--max-neurons", type=int, default=400000)
    parser.add_argument("--max-depth", type=int, default=32)
    parser.add_argument("--max-width", type=int, default=1024)
    parser.add_argument("--max-total-epochs", type=int, default=None)
    args = parser.parse_args()

    # Data
    dl_train, dl_val, dl_test = make_cifar10_loaders(
        data_root=args.data_root, batch_size=args.batch_size,
        num_workers=args.num_workers, val_split=args.val_split, download=args.download
    )

    # Model
    widths = [args.init_width] * args.init_depth
    model = AutoencoderCNN(in_ch=3, num_classes=10, widths=widths, pooling_indices=args.pool_idx)

    tcfg = TrainConfig(
        lr=args.lr, weight_decay=args.weight_decay, es_patience=args.es_patience,
        grad_clip=args.grad_clip, lambda_recon=args.lambda_recon, lambda_cls=args.lambda_cls
    )
    scfg = SearchConfig(
        delta=args.delta, patience_width=args.patience_width, patience_depth=args.patience_depth,
        ex_k=args.ex_k, max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
        max_total_epochs=args.max_total_epochs, pooling_indices=tuple(args.pool_idx)
    )

    # Search: width-only
    model = ae_search_width_only(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)

    # Final eval on test
    model.eval()
    device = tcfg.device
    model.to(device)
    import torch.nn.functional as F
    tot_r, tot_c, n = 0.0, 0.0, 0
    with torch.no_grad():
        for x, y in dl_test:
            x = x.to(device)
            y = y.to(device)
            rec, logits = model(x)
            tot_r += F.mse_loss(rec, x, reduction="sum").item()
            tot_c += F.cross_entropy(logits, y, reduction="sum").item()
            n += x.size(0)
    print(f"[TEST] recon_mse={tot_r/n:.4f}  ce={tot_c/n:.4f}  neurons={model.total_neurons()}  depth={len(model.encoder)}  widths={model.widths}")

if __name__ == "__main__":
    main()
