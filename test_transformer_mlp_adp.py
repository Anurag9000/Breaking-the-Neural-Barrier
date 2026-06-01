from __future__ import annotations

import unittest
from types import SimpleNamespace

import torch
import torch.nn as nn

from utils.transformer_mlp_adp import (
    SearchConfig,
    StagedTransformerFFNSearch,
    expand_transformer_ffn_depth,
    expand_transformer_ffn_width,
    infer_top_level_model_class,
    run_staged_transformer_ffn_search,
    transformer_ffn_architectures,
)


class Block(nn.Module):
    def __init__(self, dim: int = 4, hidden: int = 2) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 1, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.mlp(x)


class TinyTransformer(nn.Module):
    def __init__(self, blocks: int = 3) -> None:
        super().__init__()
        self.blocks = nn.ModuleList([Block() for _ in range(blocks)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for block in self.blocks:
            x = block(x)
        return x


class ShortNamedBlock(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.m = MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.m(x)


class MLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 2)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(2, 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerMLPADPTests(unittest.TestCase):
    def test_custom_ffns_expand_in_sync_without_touching_attention(self) -> None:
        model = TinyTransformer()
        attention = model.blocks[0].attn.in_proj_weight.detach().clone()

        widened = expand_transformer_ffn_width(model, ex_k=1, max_width=8)
        self.assertIsNotNone(widened)
        self.assertEqual(transformer_ffn_architectures(widened), [[3], [3], [3]])
        self.assertTrue(torch.equal(attention, widened.blocks[0].attn.in_proj_weight))

        deepened = expand_transformer_ffn_depth(widened, max_depth=4, min_new_layer_width=1)
        self.assertIsNotNone(deepened)
        self.assertEqual(transformer_ffn_architectures(deepened), [[3, 1], [3, 1], [3, 1]])
        self.assertEqual(tuple(deepened(torch.randn(2, 5, 4)).shape), (2, 5, 4))

    def test_native_encoder_decoder_ffns_expand_in_sync(self) -> None:
        model = nn.Transformer(
            d_model=4,
            nhead=1,
            num_encoder_layers=2,
            num_decoder_layers=2,
            dim_feedforward=8,
            batch_first=True,
        )
        widened = expand_transformer_ffn_width(model, ex_k=1, max_width=16)
        self.assertIsNotNone(widened)
        self.assertEqual(transformer_ffn_architectures(widened), [[9], [9], [9], [9]])

        deepened = expand_transformer_ffn_depth(widened, max_depth=3, min_new_layer_width=2)
        self.assertIsNotNone(deepened)
        self.assertEqual(transformer_ffn_architectures(deepened), [[9, 2], [9, 2], [9, 2], [9, 2]])
        output = deepened(torch.randn(2, 3, 4), torch.randn(2, 3, 4))
        self.assertEqual(tuple(output.shape), (2, 3, 4))

    def test_typed_mlp_is_discovered_when_attribute_name_is_short(self) -> None:
        model = ShortNamedBlock()
        widened = expand_transformer_ffn_width(model, ex_k=1, max_width=8)
        self.assertIsNotNone(widened)
        self.assertEqual(transformer_ffn_architectures(widened), [[3]])

    def test_staged_width_fills_each_internal_layer_before_switching_depth(self) -> None:
        model = TinyTransformer(blocks=2)
        model = expand_transformer_ffn_depth(model, max_depth=3, min_new_layer_width=1)
        self.assertIsNotNone(model)
        widened_once = expand_transformer_ffn_width(model, ex_k=1, max_width=8)
        self.assertEqual(transformer_ffn_architectures(widened_once), [[2, 2], [2, 2]])
        widened_twice = expand_transformer_ffn_width(widened_once, ex_k=1, max_width=8)
        self.assertEqual(transformer_ffn_architectures(widened_twice), [[3, 2], [3, 2]])

    def test_width_to_depth_uses_width_outer_loop_and_monotonic_global_best(self) -> None:
        config = SearchConfig(
            adp_mode="width_to_depth",
            delta=1e-6,
            width_expansion_patience=1,
            depth_expansion_patience=1,
            width_stage_margin_patience=0,
            max_width=4,
            max_depth=3,
            min_new_layer_width=1,
        )
        seen = []

        def train(candidate: nn.Module) -> float:
            seen.append(transformer_ffn_architectures(candidate)[0])
            return float(len(seen))

        best, _, search = run_staged_transformer_ffn_search(TinyTransformer(blocks=2), train, config)

        self.assertEqual(best, 1.0)
        self.assertEqual(seen[:5], [[2], [3], [3, 1], [3, 2], [3, 3]])
        self.assertIn([4, 4, 1], seen)
        self.assertEqual(search.state.candidate_index, len(seen))

    def test_all_six_modes_are_constructible(self) -> None:
        for mode in (
            "width_only",
            "depth_only",
            "alt_width",
            "alt_depth",
            "width_to_depth",
            "depth_to_width",
        ):
            search = StagedTransformerFFNSearch(TinyTransformer(), SearchConfig(adp_mode=mode))
            self.assertEqual(search.state.mode, mode)

    def test_all_six_modes_follow_staged_mlp_ordering(self) -> None:
        expected = {
            "width_only": [[2], [3]],
            "depth_only": [[2], [2, 1]],
            "alt_width": [[2], [3], [3, 1], [3, 2], [3, 3], [4, 3], [4, 4], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4]],
            "alt_depth": [[2], [2, 1], [2, 2], [3, 2], [3, 3], [3, 3, 1], [3, 3, 2], [3, 3, 3], [4, 3, 3], [4, 4, 3], [4, 4, 4]],
            "width_to_depth": [[2], [3], [3, 1], [3, 2], [3, 3], [4, 3], [4, 4], [4, 4, 1], [4, 4, 2], [4, 4, 3], [4, 4, 4]],
            "depth_to_width": [[2], [2, 1], [2, 2], [2, 2, 1], [2, 2, 2], [3, 2, 2], [3, 3, 2], [3, 3, 3], [4, 3, 3], [4, 4, 3], [4, 4, 4]],
        }
        for mode, expected_trace in expected.items():
            config = SearchConfig(
                adp_mode=mode,
                delta=1e-6,
                width_expansion_patience=1,
                depth_expansion_patience=1,
                width_stage_margin_patience=0,
                max_width=4,
                max_depth=3,
                min_new_layer_width=1,
            )
            trace = []

            def train(candidate: nn.Module) -> float:
                trace.append(transformer_ffn_architectures(candidate)[0])
                return float(len(trace))

            best, _, _ = run_staged_transformer_ffn_search(TinyTransformer(blocks=2), train, config)
            self.assertEqual(best, 1.0)
            self.assertEqual(trace, expected_trace, msg=mode)

    def test_top_level_model_inference_does_not_select_helper_blocks(self) -> None:
        baseline = SimpleNamespace(__name__=__name__, Block=Block, TinyTransformer=TinyTransformer)
        self.assertIs(infer_top_level_model_class(baseline), TinyTransformer)


if __name__ == "__main__":
    unittest.main()
