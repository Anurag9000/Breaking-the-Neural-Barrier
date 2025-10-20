import argparse
from adp_cnn_ssl_core import (AdaptiveEncoder, TrainConfig, SearchConfig, make_cifar10_ssl_loaders,
                              cnn_ssl_width_to_depth, cnn_ssl_depth_to_width,
                              cnn_ssl_alt_depth_first, cnn_ssl_alt_width_first,
                              cnn_ssl_depth_only, cnn_ssl_width_only)

def build(parser):
    parser.add_argument("--data-root", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--val-split", type=float, default=0.1)
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--two-views", action="store_true", help="If false, duplicates a single view as both.")
    parser.add_argument("--init-width", type=int, default=32)
    parser.add_argument("--init-depth", type=int, default=3)
    parser.add_argument("--proj-dim", type=int, default=None)
    parser.add_argument("--pool-idx", type=int, nargs="*", default=[0,2])
    parser.add_argument("--max-epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--es-patience", type=int, default=10)
    parser.add_argument("--grad-clip", type=float, default=None)
    parser.add_argument("--lambda-ntx", type=float, default=1.0)
    parser.add_argument("--lambda-barlow", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=0.2)
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
    enc = AdaptiveEncoder(in_ch=3, widths=widths, pooling_indices=args.pool_idx, proj_dim=args.proj_dim)
    tcfg = TrainConfig(lr=args.lr, weight_decay=args.weight_decay, es_patience=args.es_patience,
                       grad_clip=args.grad_clip, lambda_ntx=args.lambda_ntx,
                       lambda_barlow=args.lambda_barlow, temperature=args.temperature)
    scfg = SearchConfig(delta=args.delta, patience_width=args.patience_width, patience_depth=args.patience_depth,
                        ex_k=args.ex_k, max_neurons=args.max_neurons, max_depth=args.max_depth, max_width=args.max_width,
                        max_total_epochs=args.max_total_epochs, pooling_indices=tuple(args.pool_idx))
    return enc, tcfg, scfg, dl_train, dl_val, dl_test

def final_eval(enc, dl_test, device):
    import torch
    import torch.nn.functional as F
    enc.eval(); enc.to(device)
    # Evaluate SSL loss on test two-views generated from eval tf twice
    tot, n = 0.0, 0
    mean = (0.4914, 0.4822, 0.4465); std = (0.2470, 0.2435, 0.2616)
    from torchvision import transforms, datasets
    tf = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
    # Build a quick two-view iterator
    for x, _ in dl_test:
        x1 = x.to(device); x2 = x.to(device)
        with torch.no_grad():
            z1 = enc(x1); z2 = enc(x2)
            # pure cosine similarity alignment loss for report
            loss = 1 - F.cosine_similarity(z1, z2, dim=1).mean()
        tot += float(loss.item()) * x.size(0); n += x.size(0)
    print(f"[TEST] align_loss={tot/n:.4f}  neurons={enc.total_neurons()}  depth={len(enc.convs)}  widths={enc.widths}")

def main():
    parser = build(argparse.ArgumentParser())
    args = parser.parse_args()
    enc, tcfg, scfg, dl_train, dl_val, dl_test = get_common(args)
    enc = cnn_ssl_alt_width_first(enc, dl_train, dl_val, tcfg, scfg, max_epochs=args.max_epochs)
    final_eval(enc, dl_test, tcfg.device)
if __name__ == "__main__": main()
