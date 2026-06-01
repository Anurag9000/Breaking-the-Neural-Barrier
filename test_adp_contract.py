from __future__ import annotations

import copy
import unittest

import torch
import torch.nn as nn

from utils.adp_contract import run_module_adp


DEPTH_EXPANSIONS = 0


class DummyModel(nn.Module):
    def __init__(self, width: int = 4, depth: int = 1) -> None:
        super().__init__()
        self.width = int(width)
        self.depth = int(depth)
        self.weight = nn.Parameter(torch.ones(1))


class DummyConfig:
    adp_mode = "width_to_depth"
    delta = 1e-3
    patience = 2
    width_expansion_patience = 10
    depth_expansion_patience = 2
    width_stage_margin_patience = 10
    width_stage_min_improve_pct = 1.0
    ex_k = 1
    max_width = 4
    max_depth = 10
    max_neurons = 10_000
    min_new_layer_width = 1


def total_neurons(model: DummyModel, width: int, depth: int, widths=None) -> int:
    del model, widths
    return int(width) * int(depth)


def snapshot_arch_and_state(model: DummyModel, state_dict=None):
    del state_dict
    return {
        "width": int(model.width),
        "depth": int(model.depth),
        "widths": [int(model.width)] * int(model.depth),
        "model": copy.deepcopy(model),
    }


def restore_arch_and_state(model: DummyModel, snap, device=None) -> DummyModel:
    del model
    restored = copy.deepcopy(snap["model"])
    return restored.to(device) if device is not None else restored


def expand_width(model: DummyModel, ex_k: int, max_width: int, device=None, cfg=None):
    del device, cfg
    next_width = min(int(max_width), int(model.width) + int(ex_k))
    if next_width == int(model.width):
        return None
    widened = copy.deepcopy(model)
    widened.width = next_width
    return widened


def expand_depth(model: DummyModel, max_depth: int, device=None, cfg=None):
    del device, cfg
    global DEPTH_EXPANSIONS
    if int(model.depth) >= int(max_depth):
        return None
    DEPTH_EXPANSIONS += 1
    deepened = copy.deepcopy(model)
    deepened.depth += 1
    return deepened


def train_with_early_stopping(model: DummyModel, dl_train, dl_val, acfg, device, history):
    del model, dl_train, dl_val, acfg, device
    history.append(1.0)
    return 1.0, {}


class ADPContractTest(unittest.TestCase):
    def setUp(self) -> None:
        global DEPTH_EXPANSIONS
        DEPTH_EXPANSIONS = 0

    def test_width_to_depth_stops_after_depth_patience_without_improvement(self) -> None:
        best_val, best_model = run_module_adp(
            globals(),
            DummyModel(),
            dl_train=[],
            dl_val=[],
            acfg=DummyConfig(),
            device="cpu",
        )
        self.assertEqual(best_val, 1.0)
        self.assertEqual(best_model.depth, 1)
        self.assertEqual(DEPTH_EXPANSIONS, 2)


if __name__ == "__main__":
    unittest.main()
