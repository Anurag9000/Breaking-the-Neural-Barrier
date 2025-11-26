from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch

from Autoencoder.common.experiment import BaseAEExperiment, run_experiment
from Autoencoder.common.data_utils import collate_graph_batch
from Autoencoder.ae_masked.profiles import PROFILE_REGISTRY


class MaskedExperiment(BaseAEExperiment):
    variant_name = "ae_masked"
    model_builder = staticmethod(lambda meta, args: PROFILE_REGISTRY["dnn_stl"](meta, args))

    def build_model(self):
        return self.model_builder(self.meta, self.args)

    def prepare_batch(self, xb: torch.Tensor):
        data = collate_graph_batch(xb, self.meta).to(self.device)
        orig = data.x.clone()
        mask = torch.rand_like(orig).lt(self.args.mask_ratio)
        masked = orig.clone()
        masked[mask] = 0.0
        data.x_orig = orig
        data.mask = mask
        data.x = masked
        return data

    def extra_losses(self, predictions, targets, data, aux, base_metrics):
        bus_recon = aux.get("bus_recon")
        if bus_recon is None:
            return {}
        orig = data.x_orig.view(-1, self.meta.n_bus, 2)
        mask = data.mask.view(-1, self.meta.n_bus, 2)
        diff = (bus_recon - orig) * mask
        denom = mask.sum().clamp_min(1.0)
        masked_loss = diff.pow(2).sum() / denom
        return {
            "loss": self.args.masked_lambda * masked_loss,
            "masked_recon": masked_loss.detach(),
        }


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--case_name", type=str, default="pglib_opf_case118_ieee")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--alpha_cost", type=float, default=1.0)
    p.add_argument("--beta_pf", type=float, default=10.0)
    p.add_argument("--gamma_limits", type=float, default=10.0)
    p.add_argument("--eta_gen", type=float, default=5.0)
    p.add_argument("--proj_newton_steps", type=int, default=1)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
    p.add_argument("--mask_ratio", type=float, default=0.15)
    p.add_argument("--masked_lambda", type=float, default=1.0)
    p.add_argument("--alt_steps", type=str, default="1:2")
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
    MaskedExperiment.model_builder = staticmethod(builder)
    MaskedExperiment.variant_name = f"ae_masked_{args.profile}"
    run_experiment(args, MaskedExperiment)


if __name__ == "__main__":
    main()
