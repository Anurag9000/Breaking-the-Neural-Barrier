from __future__ import annotations

import types
import unittest
from unittest import mock

import MLPS.tabular.shared.dae_dnn.run_stl_parallelism_probe as probe
import MLPS.tabular.shared.dae_dnn.run_stl_ablation as stl


class ParallelismProbeTests(unittest.TestCase):
    def test_default_tasks_include_generation(self) -> None:
        self.assertIn("generation", stl.DEFAULT_TASKS)

    def test_probe_trial_sizes_increase_from_start_to_candidate_count(self) -> None:
        self.assertEqual(probe.probe_trial_sizes(2, 5), [2, 3, 4, 5])
        self.assertEqual(probe.probe_trial_sizes(2, 1), [1])
        self.assertEqual(probe.probe_trial_sizes(3, 0), [])

    def test_select_probe_candidates_takes_largest_prefix(self) -> None:
        candidates = [
            probe.ProbeCandidate("a", (9,), 900, 1),
            probe.ProbeCandidate("b", (8,), 800, 1),
            probe.ProbeCandidate("c", (7,), 700, 1),
        ]
        self.assertEqual([c.task_name for c in probe.select_probe_candidates(candidates, 2)], ["a", "b"])
        self.assertEqual([c.task_name for c in probe.select_probe_candidates(candidates, 99)], ["a", "b", "c"])

    def test_count_candidates_by_task(self) -> None:
        candidates = [
            probe.ProbeCandidate("classification", (9,), 900, 1),
            probe.ProbeCandidate("generation", (8,), 800, 1),
            probe.ProbeCandidate("generation", (7,), 700, 1),
        ]
        self.assertEqual(probe.count_candidates_by_task(candidates), {"classification": 1, "generation": 2})

    def test_parameter_matched_architectures_return_small_to_large(self) -> None:
        fake_task = types.SimpleNamespace(name="classification")
        fake_cfg = types.SimpleNamespace(
            min_width=1,
            width_step=1,
            width_count_per_depth=3,
            parameter_band=(0, 0),
            use_bn=True,
        )
        with mock.patch.object(stl, "generate_budgeted_parameter_targets", return_value=[1, 2, 3]), mock.patch.object(
            stl, "solve_parameter_matched_width", side_effect=lambda task, depth, cfg, target: int(target)
        ), mock.patch.object(stl, "_parameter_count_for_width", side_effect=lambda task, depth, width, cfg: int(width)), mock.patch.object(
            stl, "parameter_target_in_band", return_value=True
        ), mock.patch.object(
            stl, "target_parameter_count", return_value=3
        ):
            architectures = stl.parameter_matched_architectures(fake_task, 2, fake_cfg)
        self.assertEqual(architectures, [[1, 1], [2, 2], [3, 3]])


if __name__ == "__main__":
    unittest.main()
