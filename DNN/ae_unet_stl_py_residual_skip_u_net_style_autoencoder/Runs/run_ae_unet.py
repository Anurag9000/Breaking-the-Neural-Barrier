from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch

from Autoencoder.common.experiment import BaseAEExperiment, run_experiment
from Autoencoder.common.data_utils import collate_graph_batch
from Autoencoder.ae_unet_stl_py_residual_skip_u_net_style_autoencoder.profiles import PROFILE_REGISTRY


class UNetExperiment(BaseAEExperiment):
    variant_name = "ae_unet"
    model_builder = staticmethod(lambda meta, args: PROFILE_REGISTRY["dnn_stl"](meta, args))

    def build_model(self):
        return self.model_builder(self.meta, self.args)

    def prepare_batch(self, xb: torch.Tensor):
        return collate_graph_batch(xb, self.meta).to(self.device)

    def extra_losses(self, predictions, targets, data, aux, base_metrics):
        if self.args.lambda_tv <= 0:
            return {}
        features = aux.get("node_features")
        if features is None:
            return {}
        edge_index = data.edge_index
        diff = features[edge_index[0]] - features[edge_index[1]]
        tv_loss = diff.abs().mean()
        return {"loss": self.args.lambda_tv * tv_loss, "tv_loss": tv_loss.detach()}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--case_name", type=str, default="pglib_opf_case118_ieee")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--unet_depth", type=int, default=4)
    p.add_argument("--unet_width", type=int, default=128)
    p.add_argument("--graph_unet", type=int, default=1)
    p.add_argument("--lambda_tv", type=float, default=0.0)
    p.add_argument("--alpha_cost", type=float, default=1.0)
    p.add_argument("--beta_pf", type=float, default=10.0)
    p.add_argument("--gamma_limits", type=float, default=10.0)
    p.add_argument("--eta_gen", type=float, default=5.0)
    p.add_argument("--proj_newton_steps", type=int, default=1)
    p.add_argument("--train_samples", type=int, default=27000)
    p.add_argument("--val_samples", type=int, default=1500)
    p.add_argument("--test_samples", type=int, default=1500)
    p.add_argument("--batches", type=str, default="")
    p.add_argument("--profile", type=str, default="dnn_stl", choices=sorted(PROFILE_REGISTRY.keys()))
    return p


def main():
    args = build_parser().parse_args()
    args.graph_unet = bool(args.graph_unet)
    if args.profile not in PROFILE_REGISTRY:
        raise ValueError(f"Unknown profile '{args.profile}'. Available: {list(PROFILE_REGISTRY)}")
    builder = PROFILE_REGISTRY[args.profile]
    UNetExperiment.model_builder = staticmethod(builder)
    UNetExperiment.variant_name = f"ae_unet_{args.profile}"
    run_experiment(args, UNetExperiment)


if __name__ == "__main__":
    main()
