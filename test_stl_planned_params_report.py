from __future__ import annotations

import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import MLPS.tabular.shared.dae_dnn.generate_stl_planned_params_report as report


class PlannedParamsReportTests(unittest.TestCase):
    def test_summarize_task_writes_per_task_outputs(self) -> None:
        fake_task = types.SimpleNamespace(name="classification")
        fake_args = types.SimpleNamespace(param_band=(1, 3), use_bn=True, min_depth=1, max_depth=2)
        fake_cfg = types.SimpleNamespace(
            data_dir=".",
            num_workers=0,
            seed=0,
            min_width=1,
            width_step=1,
            width_count_per_depth=2,
            min_depth=1,
            max_depth=2,
            max_width=3,
            max_neurons=10_000,
            use_bn=True,
            parameter_band=(1, 3),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            output_root = Path(tmpdir)
            with mock.patch.object(report, "build_task", return_value=fake_task), mock.patch.object(
                report.stl, "REMAINING_DEPTHS_BY_TASK", {"classification": {1, 2}}
            ), mock.patch.object(report.stl, "task_depth_max_width", return_value=3), mock.patch.object(
                report.stl, "_parameter_count_for_width", side_effect=lambda task, depth, width, cfg: int(width) * int(depth)
            ), mock.patch.object(report.stl, "generate_budgeted_parameter_targets", return_value=[10, 20]), mock.patch.object(
                report.stl, "solve_parameter_matched_width", side_effect=lambda task, depth, cfg, target: 1 if int(target) == 10 else 2
            ), mock.patch.object(
                report.stl, "parameter_matched_architectures", side_effect=lambda task, depth, cfg: [[1] * depth, [2] * depth]
                ), mock.patch.object(
                    report, "candidate_parameter_count", side_effect=lambda task, architecture, use_bn: sum(int(v) for v in architecture)
                ):
                summary = report.summarize_task("classification", fake_args, output_root, fake_cfg)
                task_dir = output_root / "classification"
                self.assertTrue((task_dir / "planned_target_samples.csv").exists())
                self.assertTrue((task_dir / "planned_candidate_families.csv").exists())
                self.assertTrue((task_dir / "planned_params_decade_distribution.png").exists())
                self.assertEqual(summary["target_row_count"], 4)
                self.assertEqual(summary["candidate_row_count"], 4)


if __name__ == "__main__":
    unittest.main()
