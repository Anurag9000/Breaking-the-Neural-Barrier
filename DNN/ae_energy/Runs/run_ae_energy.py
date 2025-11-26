from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

from Autoencoder.common.experiment import BaseAEExperiment, run_experiment
from Autoencoder.common.metrics import compute_cost
from Autoencoder.ae_energy.profiles import PROFILE_REGISTRY


class EnergyExperiment(BaseAEExperiment):
    variant_name = "ae_energy"
    model_builder = staticmethod(lambda meta, args: PROFILE_REGISTRY["dnn_stl"](meta, args))

    def build_model(self):
        return self.model_builder(self.meta, self.args)

    def extra_losses(self, predictions, targets, data, aux, base_metrics):
        pg_true = targets[:, : self.meta.n_gen]
        pg_pred = aux["pg"]
        cost_true = compute_cost(pg_true, self.meta.cost_coeffs)
        cost_pred = compute_cost(pg_pred, self.meta.cost_coeffs)
        cost_penalty = self.args.alpha_cost * (cost_pred - cost_true).abs().mean()
        line_penalty = self.args.alpha_loss * (predictions - targets).pow(2).mean()
        return {"loss": cost_penalty + line_penalty, "cost_penalty": cost_penalty, "line_penalty": line_penalty}


def build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("--case_name", type=str, default="pglib_opf_case118_ieee")
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=128)
    p.add_argument("--alpha_cost", type=float, default=5.0)
    p.add_argument("--alpha_loss", type=float, default=0.1)
    p.add_argument("--beta_pf", type=float, default=10.0)
    p.add_argument("--gamma_limits", type=float, default=10.0)
    p.add_argument("--eta_gen", type=float, default=5.0)
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
    EnergyExperiment.model_builder = staticmethod(builder)
    EnergyExperiment.variant_name = f"ae_energy_{args.profile}"
    run_experiment(args, EnergyExperiment)


if __name__ == "__main__":
    main()

