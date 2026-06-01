"""
adp_base.py ─────────────────────────────────────────────────────────────────────
A thin adapter that lets legacy ADPDepth / ADPWidth / ADPPlateau code—written
for the 2018 DEN implementation—work with your modern branch-aware `DEN`.

Key duties
----------
1.  Re-export `self.layers` and `self.hidden_layers` (flat list of hidden
    `nn.Linear` layers) exactly as the old ADP helpers expect.
2.  Provide `_restore(...)` so ADP trainers can roll back after an
    unproductive expansion.
3.  Keep the legacy lists in sync whenever the network is *expanded* or
    neurons are *duplicated*.

Nothing else about `DEN` is changed; all real logic lives there.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Union, Dict

import torch.nn as nn
import torch
from dnn_den import DEN


class ADPBase_4Head(DEN):
    """Modern `DEN` + legacy symbols required by old ADP variants."""

    # --------------------------------------------------------------------- #
    # Construction & helpers                                                #
    # --------------------------------------------------------------------- #
    def __init__(self, cfg):
        super().__init__(cfg)
        self._rebuild_layers_list()  # makes self.layers & self.hidden_layers

    def _rebuild_layers_list(self) -> None:
        """
        Recreate the *flat* list of hidden layers in the exact order the
        original ADP code relied on:
            [fc1, pg_fc2, qg_fc2, va_fc2, vm_fc2]
        """
        self.layers = [self.fc1] + [getattr(self, f"{br}_fc2") for br in self.branches]
        # many ADP helpers referenced `hidden_layers` instead of `layers`
        self.hidden_layers = self.layers

    # --------------------------------------------------------------------- #
    # Legacy rollback helper                                                #
    # --------------------------------------------------------------------- #
    def _restore(self, src: Union["ADPBase_4Head", Dict[str, torch.Tensor]]) -> None:
        """
        Roll back the network **in-memory**.

        Args
        ----
        src : either another ADPBase/DEN instance *or* a `state_dict`‐like dict.
        """
        state = src.state_dict() if isinstance(src, nn.Module) else src
        # strict=False allows roll-back even if shapes grew after `src` snapshot
        self.load_state_dict(state, strict=False)
        self._rebuild_layers_list()

    # --------------------------------------------------------------------- #
    # Keep legacy lists in sync after any expansion / duplication operation #
    # --------------------------------------------------------------------- #
    def expand_layer(self, layer: nn.Linear, ex_k: int | None = None) -> nn.Linear:
        """
        Grow `layer` by `ex_k` neurons **in the output dimension** and pad the
        *next* consumer layer(s). Works for:
            • self.fc1                       (shared block)
            • getattr(self, f"{br}_fc2")     (per-branch hidden)
        """
        ex_k = ex_k or self.ex_k
        device = layer.weight.device

        # 1) --- enlarge this nn.Linear ------------------------------------
        W, b = layer.weight.data, layer.bias.data
        in_f, out_f = W.size(1), W.size(0)

        W_big = torch.cat([W, torch.zeros(ex_k, in_f, device=device)], dim=0)
        b_big = torch.cat([b, torch.zeros(ex_k,     device=device)], dim=0)

        new_layer = nn.Linear(in_f, out_f + ex_k, device=device)
        new_layer.weight.data.copy_(W_big)
        new_layer.bias.data.copy_(b_big)

        # 2) --- swap module reference on `self` ---------------------------
        if layer is self.fc1:
            self.fc1 = new_layer
            buf_prefix = "fc1"
        else:
            # identify which branch this layer belongs to
            br = next(br for br in self.branches if layer is getattr(self, f"{br}_fc2"))
            setattr(self, f"{br}_fc2", new_layer)
            buf_prefix = f"{br}_fc2"

        # 3) --- grow mask & timestamp buffers -----------------------------
        mask = getattr(self, f"{buf_prefix}_mask")
        ts   = getattr(self, f"{buf_prefix}_timestamp")
        setattr(self, f"{buf_prefix}_mask",
                torch.cat([mask, torch.ones(ex_k, dtype=torch.bool,  device=device)]))
        setattr(self, f"{buf_prefix}_timestamp",
                torch.cat([ts,   torch.full((ex_k,), self.current_task.item(),
                                            dtype=torch.long, device=device)]))

        # 4) --- pad downstream consumers ----------------------------------
        if layer is self.fc1:
            # shared layer feeds every branch-fc2
            self._pad_head_inputs(ex_k)
        else:
            self._pad_branch_head(br, ex_k)

        # 5) --- refresh legacy layer list ---------------------------------
        self._rebuild_layers_list()
        return new_layer

    def _pad_head_inputs(self, pad: int) -> None:
        """Append `pad` zero-columns to *every* branch-fc2 input weight."""
        if pad == 0:
            return
        device = self.device
        for br in self.branches:
            fc2 = getattr(self, f"{br}_fc2")
            zeros = torch.zeros(fc2.weight.size(0), pad, device=device)
            fc2.weight.data = torch.cat([fc2.weight.data, zeros], dim=1)

    def _pad_branch_head(self, br: str, pad: int) -> None:
        """When a branch hidden layer grows, pad its output head’s input."""
        if pad == 0:
            return
        head = getattr(self, f"head_{br}")
        device = head.weight.device
        zeros = torch.zeros(head.weight.size(0), pad, device=device)
        new_head = nn.Linear(head.in_features + pad, head.out_features, device=device)
        new_head.weight.data = torch.cat([head.weight.data, zeros], dim=1)
        new_head.bias.data.copy_(head.bias.data)
        setattr(self, f"head_{br}", new_head)

    # When a neuron is duplicated, `DEN` actually replaces the layer object,
    # so we just let the parent do its work and then rebuild the list.
    def _duplicate_neuron_fc1(self, idx: int) -> None:
        super()._duplicate_neuron_fc1(idx)
        self._rebuild_layers_list()

    def _duplicate_neuron_branch(self, br: str, idx: int) -> None:
        super()._duplicate_neuron_branch(br, idx)
        self._rebuild_layers_list()

    def snapshot(self) -> Dict[str, torch.Tensor]:
        """Return a *deep* copy of all parameters & buffers."""
        return deepcopy(self.state_dict())
