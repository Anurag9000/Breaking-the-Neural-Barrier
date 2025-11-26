from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch

from Autoencoder.common.experiment import BaseAEExperiment, run_experiment
from Autoencoder.common.data_utils import collate_graph_batch
from Autoencoder.ae_multi_stl_py_multi_scale_pyramid_autoencoder.profiles import PROFILE_REGISTRY


class MultiScaleExperiment(BaseAEExperiment):
    variant_name = "ae_multi_scale"
    model_builder = staticmethod(lambda meta, args: PROFILE_REGISTRY["dnn_stl"](meta, args))

    def build_model(self):
        return self.model_builder(self.meta, self.args)

    def prepare_batch(self, xb: torch.Tensor):
        return collate_graph_batch(xb, self.meta).to(self.device)

    def extra_losses(self, predictions, targets, data, aux, base_metrics):
        loss = torch.tensor(0.0, device=self.device)
        if self.args.lambda_cons > 0 and "coarse_info" in aux:
            for coarse, clusters, fine in aux["coarse_info"]:
                upsample = coarse[:, clusters.to(self.device), :]
                fine = fine.to(self.device)
                loss = loss + (upsample - fine).pow(2).mean()
            loss = self.args.lambda_cons * loss
        return {"loss": loss}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--case_name", type=str, default="pglib_opf_case118_ieee")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--levels", type=int, default=3)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lambda_cons", type=float, default=1e-3)
    p.add_argument("--interarea_lambda", type=float, default=0.0)
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
    if args.profile not in PROFILE_REGISTRY:
        raise ValueError(f"Unknown profile '{args.profile}'. Available: {list(PROFILE_REGISTRY)}")
    builder = PROFILE_REGISTRY[args.profile]
    MultiScaleExperiment.model_builder = staticmethod(builder)
    MultiScaleExperiment.variant_name = f"ae_multi_scale_{args.profile}"
    run_experiment(args, MultiScaleExperiment)


if __name__ == "__main__":
    main()
