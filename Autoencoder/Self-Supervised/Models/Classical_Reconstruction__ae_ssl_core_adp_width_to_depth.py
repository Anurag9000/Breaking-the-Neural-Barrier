import argparse
from pathlib import Path
import importlib.util

import torch

# Load baseline module
BASE_PATH = Path(__file__).resolve().with_name("Classical_Reconstruction__ae_ssl_core.py")
_spec = importlib.util.spec_from_file_location("ae_core", BASE_PATH)
ae_core = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ae_core)

AutoencoderSSL = ae_core.AutoencoderSSL  # type: ignore
TrainConfig = ae_core.TrainConfig  # type: ignore
SearchConfig = ae_core.SearchConfig  # type: ignore
make_cifar10_ssl_loaders = ae_core.make_cifar10_ssl_loaders  # type: ignore


def run_adp(adp_mode: str, args):
    # Data
    dl_train, dl_val, _ = make_cifar10_ssl_loaders(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        val_split=args.val_split,
        download=True,
        seed=args.seed,
        two_views=args.two_views,
    )
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))

    # Model + configs
    model = AutoencoderSSL(
        in_ch=3,
        widths=args.widths,
        pooling_indices=args.pool_idx,
        bias=True,
        proj_dim=args.projector_dim,
    ).to(device)
    tcfg = TrainConfig(
        lr=args.lr,
        weight_decay=args.weight_decay,
        es_patience=args.es_patience,
        grad_clip=args.grad_clip,
        lambda_recon=args.lambda_recon,
        lambda_consistency=args.lambda_consistency,
        lambda_barlow=args.lambda_barlow,
        projector_dim=args.projector_dim,
        two_views=args.two_views,
        device=device.type,
    )
    scfg = SearchConfig(
        delta=args.delta,
        patience_width=args.patience_width,
        patience_depth=args.patience_depth,
        ex_k=args.ex_k,
        max_neurons=args.max_neurons,
        max_depth=args.max_depth,
        max_width=args.max_width,
        max_total_epochs=args.max_total_epochs,
        pooling_indices=tuple(args.pool_idx),
    )

    # Dispatch
    if adp_mode == "width_only":
        ae_core.ae_ssl_width_only(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.inner_epochs,
                                  log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    elif adp_mode == "depth_only":
        ae_core.ae_ssl_depth_only(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.inner_epochs,
                                  log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    elif adp_mode == "width_to_depth":
        ae_core.ae_ssl_width_to_depth(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.inner_epochs,
                                      log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    elif adp_mode == "depth_to_width":
        ae_core.ae_ssl_depth_to_width(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.inner_epochs,
                                      log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    elif adp_mode == "alt_width":
        ae_core.ae_ssl_alt_width_first(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.inner_epochs,
                                       log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    elif adp_mode == "alt_depth":
        ae_core.ae_ssl_alt_depth_first(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.inner_epochs,
                                       log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    elif adp_mode == "width":
        ae_core.ae_ssl_width_only(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.inner_epochs,
                                  log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    elif adp_mode == "depth":
        ae_core.ae_ssl_depth_only(model, dl_train, dl_val, tcfg, scfg, max_epochs=args.inner_epochs,
                                  log_loss=args.plot_loss, log_neurons=args.plot_neurons, results_dir=args.results_dir)
    else:
        raise ValueError(f"Unsupported adp_mode {adp_mode}")
    # Model is updated in-place; nothing else to return
    return model


def main():
    p = argparse.ArgumentParser(description="ADP Classical Reconstruction AE (SSL) width/depth search")
    p.add_argument("--data-root", type=str, default="./data")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--num-workers", type=int, default=4)
    p.add_argument("--val-split", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--two-views", action="store_true", default=True)
    p.add_argument("--widths", type=int, nargs="+", default=[32, 64, 128])
    p.add_argument("--pool-idx", type=int, nargs="*", default=[0, 2])
    p.add_argument("--projector-dim", type=int, default=None)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--es-patience", type=int, default=20)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--lambda-recon", type=float, default=1.0)
    p.add_argument("--lambda-consistency", type=float, default=1.0)
    p.add_argument("--lambda-barlow", type=float, default=0.0)
    p.add_argument("--delta", type=float, default=1e-3)
    p.add_argument("--patience-width", type=int, default=100000000)
    p.add_argument("--patience-depth", type=int, default=100000000)
    p.add_argument("--ex-k", type=int, default=8)
    p.add_argument("--max-neurons", type=int, default=1_000_000)
    p.add_argument("--max-depth", type=int, default=32)
    p.add_argument("--max-width", type=int, default=1024)
    p.add_argument("--max-total-epochs", type=int, default=None)
    p.add_argument("--inner-epochs", type=int, default=50)
    p.add_argument("--adp-mode", type=str, default="width_to_depth",
                   choices=["width_only", "depth_only", "width_to_depth", "depth_to_width", "alt_width", "alt_depth", "width", "depth"])
    p.add_argument("--device", type=str, default=None, help="force device, e.g. cpu or cuda")
    p.add_argument("--results-dir", type=Path, default=Path("results_adp_classical_ssl"))
    p.add_argument("--plot-loss", action="store_true", help="Save loss-vs-epoch (log scale)")
    p.add_argument("--plot-neurons", action="store_true", help="Save neurons-vs-loss (log scale)")
    args = p.parse_args()

    model = run_adp(args.adp_mode, args)
    print(f"[ADP Classical AE SSL] mode={args.adp_mode} widths={model.widths} depth={len(model.widths)}")


if __name__ == "__main__":
    main()
