from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[3]))

import torch
import torch.nn.functional as F

from Autoencoder.common.experiment import BaseAEExperiment, run_experiment
from Autoencoder.ae_robust.profiles import PROFILE_REGISTRY
from Dyn_DNN4OPF.utils.constraint_losses import power_balance_residuals


class RobustExperiment(BaseAEExperiment):
    variant_name = "ae_robust"
    model_builder = staticmethod(lambda meta, args: PROFILE_REGISTRY["dnn_stl"](meta, args))

    def build_model(self):
        return self.model_builder(self.meta, self.args)

    def extra_losses(self, predictions, targets, data, aux, base_metrics):
        pd = data.pdqd[:, : self.meta.n_bus]
        qd = data.pdqd[:, self.meta.n_bus :]
        residual_p, residual_q = power_balance_residuals(
            predictions,
            pd,
            qd,
            self.model.y_bus_real.to(self.device),
            self.model.y_bus_imag.to(self.device),
            self.model.gen_bus_idx.to(self.device),
            self.model.load_bus_idx.to(self.device),
        )
        pf_huber = F.smooth_l1_loss(residual_p, torch.zeros_like(residual_p)) + F.smooth_l1_loss(
            residual_q, torch.zeros_like(residual_q)
        )
        over = torch.clamp(predictions - self.bounds_hi.to(self.device), min=0)
        under = torch.clamp(self.bounds_lo.to(self.device) - predictions, min=0)
        limit_huber = F.smooth_l1_loss(over + under, torch.zeros_like(over))
        replacement = self.args.beta_pf * pf_huber + self.args.gamma_limits * limit_huber
        base_terms = self.args.beta_pf * base_metrics["pf_rms"] + self.args.gamma_limits * base_metrics["limits"]
        return {"loss": replacement - base_terms, "pf_huber": pf_huber.detach(), "limit_huber": limit_huber.detach()}


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
    p.add_argument("--robust_delta", type=float, default=0.05)
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
    RobustExperiment.model_builder = staticmethod(builder)
    RobustExperiment.variant_name = f"ae_robust_{args.profile}"
    run_experiment(args, RobustExperiment)


if __name__ == "__main__":
    main()
