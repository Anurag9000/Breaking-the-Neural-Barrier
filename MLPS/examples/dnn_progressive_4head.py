"""
===============================================================================
Progressive Neural Network (PNN) Implementation
===============================================================================

This module implements the Progressive Neural Network architecture, as introduced by 
Rusu et al., 2016, designed for continual learning by **adding a new network column** per 
task and leveraging lateral connections for transfer while preventing forgetting.

--------------------------------------------------------------------------------
Key Contributions:
    1. **No forgetting by construction:**
       Each task is assigned a separate frozen column network; prior knowledge is retained 
       by freezing parameters of earlier columns.
       
    2. **Transfer via lateral adapters:**
        Lateral connections from previous columns feed into each layer of the new column, 
        enabling feature reuse and accelerating learning.
       
    3. **Adapter modules with learnable scalar multipliers:**
       Non-linear lateral connections implemented as lightweight MLPs scaled by learned α parameters.
       
    4. **Fine-grained interpretability:**
       Computes Average Fisher Sensitivity (AFS) and Average Perturbation Sensitivity (APS) to 
       quantify transfer strength from prior columns.

--------------------------------------------------------------------------------
Model Structure:
    - Each **column** is a fully connected feedforward network with:
        • Two hidden layers with ReLU
        • One output layer
    - Each new task creates a new column with lateral adapters from all previous columns.
    - Adapter weights and α scalars modulate lateral influence.
    - Prior columns' weights are frozen to prevent forgetting.

--------------------------------------------------------------------------------
Training Workflow:
    1. Initialize first column; train on first task conventionally.
    2. For each new task:
        - Freeze previous columns.
        - Add new column with fresh weights.
        - Add lateral adapters connecting previous columns to new column.
        - Train new column + adapters only.
    3. After each task:
        - Prune adapters with low α or weight magnitude.
        - Compute AFS/APS scores for transfer analysis.
        - Identify dead neurons for potential pruning.

--------------------------------------------------------------------------------
Evaluation and Interpretation:
    - Uses Fisher Information-based sensitivity metrics (AFS) to measure layerwise transfer.
    - APS computed via perturbation analysis injecting noise in lateral inputs.
    - Dead neurons identified as those with near-zero activations during validation.

--------------------------------------------------------------------------------
Integration:
    - Used in sequential training scenarios in `trainer.py`.
    - Adapter pruning and diagnostics facilitated by helper functions.
    - Enables explicit transfer and no-forgetting guarantees.

--------------------------------------------------------------------------------
References:
    Andrei A. Rusu et al., "Progressive Neural Networks," arXiv 2016.
"""
import torch
import torch.nn as nn
import logging
from typing import Optional, List, Dict, Tuple, Set
import torch.nn.functional as F
import itertools
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _slice(dims: List[int], idx: int) -> Tuple[int, int]:
    """Return (start, end) slice indices for *idx* within concatenated dims."""
    s = sum(dims[:idx])
    return s, s + dims[idx]


class AlphaAdapter(nn.Module):
    """Two-layer bottleneck MLP scaled by a learnable scalar α (GPU-first)."""

    def __init__(self, hidden_dim: int, reduction: int = 4) -> None:  # noqa: D401
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1, device=device).uniform_(-0.5, 0.5))
        red = hidden_dim // reduction
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, red),
            nn.ReLU(),
            nn.Linear(red, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        return self.adapter(self.alpha * x)


class ProgressiveColumn(nn.Module):
    """Feed-forward column for a Progressive Neural Net (GPU-first).

    Architecture
    ------------
    * FC(hidden_dim) → ReLU → FC(hidden_dim) → ReLU → FC(output_dim)
    * Final bounded activation via :class:`BoundedAct` (no opt-out).

    Parameters
    ----------
    input_dim : int
        Dimension of input features.
    hidden_dim : int
        Width of each hidden layer.
    output_dim : int
        Dimension of the task-specific output vector.
    bounds_low / bounds_high : torch.Tensor
        Element-wise lower / upper bounds (sliced for this task).
    mask : torch.Tensor
        Binary mask (same length as *output_dim*) selecting which outputs are
        subject to bounding.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        *,
        bounds_low: torch.Tensor,
        bounds_high: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        super().__init__()

        # core layers -----------------------------------------------------
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        # bounded activation (mandatory) ----------------------------------
        self.bound_layer: nn.Module = BoundedAct(
            bounds_low.to(device),
            bounds_high.to(device),
            mask.to(device),
        )

        self.to(device)

    # --------------------------------------------------------------------
    def forward_with_activations(self, x: torch.Tensor):
        """Forward pass returning intermediate activations for adapters."""
        x = x.to(device, non_blocking=True)
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        out = self.bound_layer(self.output_layer(h2))
        return h1, h2, out

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # noqa: D401
        """Standard forward pass (no lateral outputs)."""
        x = x.to(device, non_blocking=True)
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        return self.bound_layer(self.output_layer(h2))


class DNN_Progressive_4HEAD(nn.Module):
    """Progressive Neural Net: **four static columns** (Pg, Qg, Va, Vm)."""

    HEADS = ("pg", "qg", "va", "vm")

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        *,
        bounds_low: torch.Tensor,
        bounds_high: torch.Tensor,
        mask: torch.Tensor,
    ) -> None:
        super().__init__()

        if hidden_dim is None:
            hidden_dim = 4 * input_dim
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        # ── derive per-task output dims from bounds ----------------------
        total_out = len(bounds_low)
        n_bus = input_dim // 2
        n_gen = (total_out - 2 * n_bus) // 2
        assert total_out == 2 * n_gen + 2 * n_bus, "Inconsistent bound lengths"
        self.output_dims = [n_gen, n_gen, n_bus, n_bus]

        # move bounds/mask to GPU once -----------------------------------
        bounds_low = bounds_low.to(device)
        bounds_high = bounds_high.to(device)
        mask = mask.to(device)

        # ── build columns & adapters ------------------------------------
        self.columns = nn.ModuleList()
        self.adapters = []  # list[(ad1, ad2)] per column (kept for indexing)

        for col_id, odim in enumerate(self.output_dims):
            s, e = _slice(self.output_dims, col_id)
            self.columns.append(
                ProgressiveColumn(
                    input_dim,
                    hidden_dim,
                    odim,
                    bounds_low=bounds_low[s:e],
                    bounds_high=bounds_high[s:e],
                    mask=mask[s:e],
                )
            )

            if col_id == 0:
                # register empty ModuleLists so parameters() sees them
                setattr(self, f"_ad1_{col_id}", nn.ModuleList())
                setattr(self, f"_ad2_{col_id}", nn.ModuleList())
            else:
                ad1 = nn.ModuleList([AlphaAdapter(hidden_dim).to(device) for _ in range(col_id)])
                ad2 = nn.ModuleList([AlphaAdapter(hidden_dim).to(device) for _ in range(col_id)])
                # register as attributes to ensure optimizer sees them
                setattr(self, f"_ad1_{col_id}", ad1)
                setattr(self, f"_ad2_{col_id}", ad2)

                # freeze earlier columns --------------------------------
                for prev in self.columns[:col_id]:
                    for p in prev.parameters():
                        p.requires_grad = False

            # keep tuple view for existing code
            self.adapters.append(
                (getattr(self, f"_ad1_{col_id}"), getattr(self, f"_ad2_{col_id}"))
            )

        self.to(device)

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:  # noqa: D401
        """Forward pass for *task_id* using lateral adapters."""
        if not (0 <= task_id < 4):
            raise IndexError("task_id must be 0≤id<4 (Pg, Qg, Va, Vm)")

        x = x.to(device, non_blocking=True)
        if task_id == 0:
            return self.columns[0](x)

        # gather lateral activations -------------------------------------
        h1_lat, h2_lat = [], []
        for j in range(task_id):
            h1_j, h2_j, _ = self.columns[j].forward_with_activations(x)
            h1_lat.append(h1_j)
            h2_lat.append(h2_j)

        a1, a2 = self.adapters[task_id]
        h1_sum = sum(a1[j](h1_lat[j]) for j in range(task_id))
        h2_sum = sum(a2[j](h2_lat[j]) for j in range(task_id))

        # forward through target column with lateral input ----------------
        col = self.columns[task_id]
        h1 = col.act1(col.fc1(x) + h1_sum)
        h2 = col.act2(col.fc2(h1) + h2_sum)
        return col.bound_layer(col.output_layer(h2))

    def get_all_shared_weights(self) -> List[nn.Parameter]:  # noqa: D401
        """Return *trainable* parameters from columns & adapters (GPU-ready)."""
        weights = [p for col in self.columns for p in col.parameters() if p.requires_grad]
        for ad_pair in self.adapters:
            if ad_pair[0] is not None:
                weights.extend(
                    p for p in itertools.chain(ad_pair[0].parameters(), ad_pair[1].parameters())
                    if p.requires_grad
                )
        return weights

    def prune_adapters(self, threshold: float = 1e-3) -> None:  # noqa: D401
        """Hard-zero adapter weights whose magnitude < *threshold*."""
        for task_id, (ad1, ad2) in enumerate(self.adapters):
            if ad1 is None:
                continue  # column-0 has no adapters
            for j, adapter in enumerate(itertools.chain(ad1, ad2)):
                for name, param in adapter.adapter.named_parameters():
                    mask = param.abs() >= threshold
                    num_pruned = torch.count_nonzero(~mask).item()
                    param.data.mul_(mask)  # in-place masking
                    logger.debug("Pruned %d weights in adapter[%d][%d].%s", num_pruned, task_id, j, name)

    def prune_small_alpha_adapters(self, alpha_threshold: float = 1e-3) -> None:  # noqa: D401
        """Zero-out entire adapter if |alpha| < *alpha_threshold*."""
        for ad1, ad2 in self.adapters:
            for layer_adapters in (ad1, ad2):
                if layer_adapters is None:
                    continue
                for adapter in layer_adapters:
                    if adapter.alpha.abs().item() < alpha_threshold:
                        for param in adapter.parameters():
                            param.data.zero_()

    def log_adapter_sparsity(self) -> None:  # noqa: D401
        """Log % of non-zero weights remaining in each adapter set."""
        for task_id, (ad1, ad2) in enumerate(self.adapters):
            if ad1 is None:
                continue
            total = kept = 0
            for adapter in itertools.chain(ad1, ad2):
                for param in adapter.adapter.parameters():
                    total += param.numel()
                    kept += torch.count_nonzero(param).item()
            pct = 100.0 * kept / max(total, 1)
            logger.info("Task %d adapters: %d/%d weights kept (%.2f%%)", task_id, kept, total, pct)

    def compute_afs(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> Dict[str, List[float]]:  # noqa: D401
        """Compute AFS for lateral inputs into *task_id* column (regression)."""
        device = next(self.parameters()).device
        self.eval()
        afs = {"layer1": [0.0] * task_id, "layer2": [0.0] * task_id}
        loss_fn = nn.MSELoss(reduction="mean")
        count = 0

        for batch in dataloader:
            xb, yb = batch[:2]
            xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

            h1_inputs, h2_inputs = [], []
            for j in range(task_id):
                h1_j, h2_j, _ = self.columns[j].forward_with_activations(xb)
                h1_j.retain_grad()
                h2_j.retain_grad()
                h1_inputs.append(h1_j)
                h2_inputs.append(h2_j)

            # lateral sums ----------------------------------------------------
            ad1, ad2 = self.adapters[task_id]
            h1_lat = sum(ad1[j](h1_inputs[j]) for j in range(task_id))
            h2_lat = sum(ad2[j](h2_inputs[j]) for j in range(task_id))

            # forward through target column ----------------------------------
            col = self.columns[task_id]
            h1 = col.act1(col.fc1(xb) + h1_lat)
            h2 = col.act2(col.fc2(h1) + h2_lat)
            out = col.bound_layer(col.output_layer(h2))
            loss = loss_fn(out, yb)

            self.zero_grad(set_to_none=True)
            loss.backward()

            # accumulate squared-grad norms ----------------------------------
            for j in range(task_id):
                afs["layer1"][j] += h1_inputs[j].grad.norm(2, dim=1).pow(2).mean().item()
                afs["layer2"][j] += h2_inputs[j].grad.norm(2, dim=1).pow(2).mean().item()
            count += 1

        if count:
            for key in ("layer1", "layer2"):
                afs[key] = [v / count for v in afs[key]]
        return afs

    def compute_aps(self, task_id: int, dataloader: torch.utils.data.DataLoader) -> Dict[str, List[float]]:  # noqa: D401
        """Compute APS = 1/σ* where σ* halves performance."""
        device = next(self.parameters()).device
        self.eval()
        aps = {"layer1": [], "layer2": []}

        base_mse = self.evaluate_model(task_id, dataloader)

        for layer_idx, layer_name in enumerate(("layer1", "layer2")):
            for j in range(task_id):
                sigma_star = self.find_critical_sigma(task_id, layer_idx, j, dataloader, base_mse)
                aps[layer_name].append(float("inf") if sigma_star == 0 else 1.0 / sigma_star)
        return aps

    def identify_dead_neurons(self, column_id: int, dataloader, eps: float = 1e-5) -> Dict[int, Set[int]]:  # noqa: D401
        """Return indices of neurons whose |activation| < eps for all samples."""
        device = next(self.parameters()).device
        self.eval()
        dead = {1: set(), 2: set()}
        max1 = max2 = None

        with torch.no_grad():
            for xb, *_ in dataloader:
                xb = xb.to(device, non_blocking=True)
                h1, h2, _ = self.columns[column_id].forward_with_activations(xb)
                max1 = torch.maximum(h1.abs().max(dim=0).values, max1) if max1 is not None else h1.abs().max(dim=0).values
                max2 = torch.maximum(h2.abs().max(dim=0).values, max2) if max2 is not None else h2.abs().max(dim=0).values

        dead[1] = set(torch.nonzero(max1 < eps, as_tuple=True)[0].tolist())
        dead[2] = set(torch.nonzero(max2 < eps, as_tuple=True)[0].tolist())
        return dead

    def find_critical_sigma(self, task_id: int, layer_idx: int, column_j: int, dataloader, base_mse: float, tol: float = 0.05) -> float:  # noqa: D401
        """Return smallest σ where MSE ≥ 2× base (≈50% perf drop)."""
        low, high = 1e-4, 1.0
        mid = low
        for _ in range(10):
            mid = (low + high) / 2
            mse = self.evaluate_with_noise(task_id, layer_idx, column_j, dataloader, mid)
            if mse >= 2.0 * base_mse:
                high = mid
            else:
                low = mid
            if abs(mse - 2.0 * base_mse) < tol * base_mse:
                break
        return mid

    def evaluate_model(self, task_id: int, dataloader) -> float:  # noqa: D401
        device = next(self.parameters()).device
        self.eval()
        total, elems = 0.0, 0
        with torch.no_grad():
            for xb, yb, *_ in dataloader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
                preds = self.forward(xb, task_id)
                total += F.mse_loss(preds, yb, reduction="sum").item()
                elems += yb.numel()
        return total / elems if elems else 0.0

    def evaluate_with_noise(self, task_id: int, layer_idx: int, column_j: int, dataloader, sigma: float) -> float:  # noqa: D401
        """
        Inject Gaussian noise (std=sigma) into the *lateral input* originating
        from source column_j at the specified layer (0 or 1), and measure MSE.
        """
        device = next(self.parameters()).device
        self.eval()
        total, elems = 0.0, 0

        with torch.no_grad():
            for xb, yb, *_ in dataloader:
                xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)

                # features from ALL source columns 0..task_id-1 -------------
                h1_inputs, h2_inputs = [], []
                for k in range(task_id):
                    h1_k, h2_k, _ = self.columns[k].forward_with_activations(xb)
                    h1_inputs.append(h1_k)
                    h2_inputs.append(h2_k)

                # perturb chosen source column_j at the chosen layer ---------
                feat_j = h1_inputs[column_j] if layer_idx == 0 else h2_inputs[column_j]
                noise = sigma * torch.randn_like(feat_j)
                perturbed = self.adapters[task_id][layer_idx][column_j](feat_j + noise)

                # sum of *other* adapters (unperturbed) ----------------------
                others = sum(
                    self.adapters[task_id][layer_idx][k](
                        h1_inputs[k] if layer_idx == 0 else h2_inputs[k]
                    )
                    for k in range(task_id) if k != column_j
                )
                lateral_sum = perturbed + others

                # forward through target column with the lateral injection ---
                col = self.columns[task_id]
                if layer_idx == 0:
                    h1 = col.act1(col.fc1(xb) + lateral_sum)
                    h2 = col.act2(col.fc2(h1))
                else:
                    h1 = col.act1(col.fc1(xb))
                    h2 = col.act2(col.fc2(h1) + lateral_sum)

                out = col.bound_layer(col.output_layer(h2))

                total += F.mse_loss(out, yb, reduction="sum").item()
                elems += yb.numel()

        return total / elems if elems else 0.0
