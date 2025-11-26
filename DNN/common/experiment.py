"""Experiment orchestration for autoencoder variants."""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import time

import torch
from torch.utils.data import DataLoader

from Dyn_DNN4OPF.data.opf_loader import get_data_loaders, load_output_bounds
from Dyn_DNN4OPF.utils.config import get_io_dims_from_loader, default_mask, check_bounds_compatibility
from Dyn_DNN4OPF.utils.logger_plotter import save_logs_to_csv
from Dyn_DNN4OPF.utils.repro import set_determinism

import torch_geometric

from .core import CompositeLoss, CompositeLossConfig
from .data_utils import CaseMetadata, collate_graph_batch
from .metrics import summarise_metrics


class BaseAEExperiment:
    variant_name: str = "base"

    def __init__(self, args):
        self.args = args
        set_determinism(42)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.meta = CaseMetadata(args.case_name)
        batches = None
        if getattr(args, "batches", ""):
            batches = [int(x) for x in args.batches.split(",") if x]
        self.train_loader, self.val_loader, self.test_loader, self.obj_test = get_data_loaders(
            args.batch_size,
            args.case_name,
            args.train_samples,
            args.val_samples,
            args.test_samples,
            batches,
        )
        self.in_dim, self.out_dim = get_io_dims_from_loader(self.train_loader)
        self.n_bus = self.in_dim // 2
        self.n_gen = self.out_dim // 2 - self.n_bus
        self.bounds_lo, self.bounds_hi = load_output_bounds(args.case_name)
        mask = default_mask(self.n_gen, self.n_bus)
        check_bounds_compatibility(self.bounds_lo, self.bounds_hi, mask, self.out_dim)
        self.mask = mask.to(torch.bool)

        self.model = self.build_model().to(self.device)
        self.loss_core = CompositeLoss(
            CompositeLossConfig(
                alpha_cost=args.alpha_cost,
                beta_pf=args.beta_pf,
                gamma_limits=args.gamma_limits,
                eta_gen=args.eta_gen,
                robust_delta=getattr(args, "robust_delta", None),
            )
        )
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=args.lr)
        self.logs = []
        self.best_val = float("inf")
        self.patience_left = args.patience

        ts = time.strftime("%Y%m%d_%H%M%S")
        root = Path.cwd() / "Results" / self.variant_name / args.case_name / ts
        for sub in ("models", "logs", "plots", "diagnostics"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        self.output_dir = root
        self.log_path = root / "logs" / "train.csv"
        self.best_model_path = root / "models" / "best.pt"

    # Hooks -----------------------------------------------------------------
    def build_model(self) -> torch.nn.Module:
        raise NotImplementedError

    def prepare_batch(self, xb: torch.Tensor) -> torch_geometric.data.Data:  # type: ignore
        return collate_graph_batch(xb, self.meta).to(self.device)

    def extra_losses(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        data,
        aux: Optional[Dict] = None,
        base_metrics: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        return {}

    def post_batch(self, *args, **kwargs) -> None:
        pass

    def train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        batches = 0
        for xb, yb, objb in self.train_loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            data = self.prepare_batch(xb)
            predictions, aux = self.model(data)
            base_metrics = self.loss_core(
                predictions,
                yb,
                xb,
                self.bounds_lo.to(self.device),
                self.bounds_hi.to(self.device),
                self.model.y_bus_real.to(self.device),
                self.model.y_bus_imag.to(self.device),
                self.model.gen_bus_idx.to(self.device),
                self.model.load_bus_idx.to(self.device),
            )
            extras = self.extra_losses(predictions, yb, data, aux, base_metrics)
            loss = base_metrics["loss"] + extras.get("loss", 0.0)
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
            batches += 1
            self.post_batch(predictions, yb, data, aux)
        return total_loss / max(1, batches)

    @torch.no_grad()
    def evaluate(self, loader: DataLoader) -> Tuple[float, Dict[str, float]]:
        self.model.eval()
        total = 0.0
        batches = 0
        metric_accum: Dict[str, float] = {"pf_rms": 0.0, "violations": 0.0, "cost_gap": 0.0, "max_overload": 0.0, "max_v_bounds": 0.0}
        for xb, yb, objb in loader:
            xb = xb.to(self.device)
            yb = yb.to(self.device)
            data = self.prepare_batch(xb)
            predictions, aux = self.model(data)
            base_metrics = self.loss_core(
                predictions,
                yb,
                xb,
                self.bounds_lo.to(self.device),
                self.bounds_hi.to(self.device),
                self.model.y_bus_real.to(self.device),
                self.model.y_bus_imag.to(self.device),
                self.model.gen_bus_idx.to(self.device),
                self.model.load_bus_idx.to(self.device),
            )
            extras = self.extra_losses(predictions, yb, data, aux, base_metrics)
            loss = base_metrics["loss"] + extras.get("loss", 0.0)
            total += loss.item()
            batches += 1
            metrics = summarise_metrics(
                predictions,
                yb,
                xb,
                self.bounds_lo.to(self.device),
                self.bounds_hi.to(self.device),
                self.model.y_bus_real.to(self.device),
                self.model.y_bus_imag.to(self.device),
                self.model.gen_bus_idx.to(self.device),
                self.model.load_bus_idx.to(self.device),
                self.meta.cost_coeffs,
            )
            for k in metric_accum:
                metric_accum[k] += metrics[k]
        avg_loss = total / max(1, batches)
        metrics = {k: v / max(1, batches) for k, v in metric_accum.items()}
        return avg_loss, metrics

    def run(self) -> None:
        for epoch in range(1, self.args.epochs + 1):
            train_loss = self.train_epoch()
            val_loss, val_metrics = self.evaluate(self.val_loader)
            self.logs.append(
                {
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    **{f"val_{k}": v for k, v in val_metrics.items()},
                }
            )
            if val_loss < self.best_val - 1e-6:
                self.best_val = val_loss
                self.patience_left = self.args.patience
                torch.save(self.model.state_dict(), self.best_model_path)
            else:
                self.patience_left -= 1
            if epoch % 5 == 0:
                save_logs_to_csv(self.logs, str(self.log_path))
            if self.patience_left <= 0:
                break
        save_logs_to_csv(self.logs, str(self.log_path))

        best_state = torch.load(self.best_model_path, map_location=self.device)
        self.model.load_state_dict(best_state)
        test_loss, test_metrics = self.evaluate(self.test_loader)
        summary_path = self.output_dir / "logs" / "test_summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"test_loss={test_loss}\n")
            for k, v in test_metrics.items():
                f.write(f"{k}={v}\n")


def run_experiment(args, experiment_cls):
    exp = experiment_cls(args)
    exp.run()

