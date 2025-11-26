from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from Autoencoder.common.experiment import BaseAEExperiment, run_experiment
from Autoencoder.ae_lowrank_stl_py_low_rank_nuclear_norm_regularized_ae.profiles import PROFILE_REGISTRY


class LowRankExperiment(BaseAEExperiment):
    variant_name = "ae_lowrank_stl"
    model_builder = staticmethod(lambda meta, args: PROFILE_REGISTRY["dnn_stl"](meta, args))

    def build_model(self):
        return self.model_builder(self.meta, self.args)

    def extra_losses(self, predictions, targets, data, aux, base_metrics):
        u_batch = aux.get("u_batch")
        if u_batch is None:
            return {}
        penalty = self.args.nuclear_lambda * (
            u_batch.pow(2).mean() + self.model.latent_factor.pow(2).mean()
        )
        return {"loss": penalty, "lowrank_penalty": penalty}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--case_name", type=str, default="pglib_opf_case118_ieee")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--rank_k", type=int, default=32)
    p.add_argument("--nuclear_lambda", type=float, default=1e-2)
    p.add_argument("--beta_pf", type=float, default=10.0)
    p.add_argument("--gamma_limits", type=float, default=10.0)
    p.add_argument("--eta_gen", type=float, default=5.0)
    p.add_argument("--alpha_cost", type=float, default=1.0)
    p.add_argument("--proj_newton_steps", type=int, default=1)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--layers", type=int, default=3)
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
    LowRankExperiment.model_builder = staticmethod(builder)
    LowRankExperiment.variant_name = f"ae_lowrank_{args.profile}"
    run_experiment(args, LowRankExperiment)


if __name__ == "__main__":
    main()
