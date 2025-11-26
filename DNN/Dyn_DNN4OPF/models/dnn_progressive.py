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
from typing import Optional, List, Dict
import torch.nn.functional as F
import itertools
from Dyn_DNN4OPF.utils.bounded_act import BoundedAct

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)

class AlphaAdapter(nn.Module):
    """
    Adapter block with learnable scalar α multiplying a 2-layer MLP.
    """
    def __init__(self, hidden_dim: int, reduction: int = 4) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.empty(1).uniform_(-0.5, 0.5))
        reduced_dim = hidden_dim // reduction
        self.adapter = nn.Sequential(
            nn.Linear(hidden_dim, reduced_dim),  # V_i projection
            nn.ReLU(),
            nn.Linear(reduced_dim, hidden_dim)   # U_i expansion
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.adapter(self.alpha * x)

class ProgressiveColumn(nn.Module):
    """
    Single feedforward column used in progressive neural networks.
    Consists of two hidden layers and one output layer, all fully connected.
    """
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 use_bounds: bool = False, bounds_low=None, bounds_high=None, mask=None) -> None:
        """
        Initialize a progressive column.

        Args:
            input_dim (int): Dimension of input features.
            hidden_dim (int): Number of neurons in each hidden layer.
            output_dim (int): Dimension of output.
            use_bounds (bool): Whether to apply bounded output activation.
            bounds_low (torch.Tensor): Lower bound for each output node.
            bounds_high (torch.Tensor): Upper bound for each output node.
            mask (torch.Tensor): Mask indicating which outputs apply bounds.
        """
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.act2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_dim, output_dim)

        if use_bounds:
            if any(arg is None for arg in (bounds_low, bounds_high, mask)):
                raise ValueError("Bounds or mask must be provided when use_bounds=True")
            self.bound_layer = BoundedAct(bounds_low, bounds_high, mask)
            if not (len(bounds_low) == len(bounds_high) == len(mask) == output_dim):
                raise ValueError(
                    f"Bounds/mask length must match output_dim={output_dim}. "
                    f"Got: {len(bounds_low)}, {len(bounds_high)}, {len(mask)}"
                )

        else:
            self.bound_layer = nn.Identity()

    def forward_with_activations(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass that returns intermediate activations for lateral connections.

        Returns:
            h1: Activation after first hidden layer.
            h2: Activation after second hidden layer.
            out: Final output (post-bound if enabled).
        """
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        out = self.bound_layer(self.output_layer(h2))
        return h1, h2, out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Standard forward pass (no lateral outputs).
        """
        h1 = self.act1(self.fc1(x))
        h2 = self.act2(self.fc2(h1))
        return self.bound_layer(self.output_layer(h2))

class DNN_Progressive(nn.Module):
    """
    Progressive Neural Network with multiple columns and layer-wise lateral adapters.
    Implements fine-grained transfer learning through lateral connections at each hidden layer.
    """
    def __init__(
        self,
        input_dim: int,
        hidden_dim: Optional[int] = None,
        *,
        use_bounds: bool = False,
        bounds_low: Optional[torch.Tensor] = None,
        bounds_high: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> None:
        """
        Initialize the progressive network.

        Args:
            input_dim (int): Input feature dimension.
            hidden_dim (Optional[int]): Hidden layer dimension;
                defaults to 4 * input_dim if not provided.
            use_bounds (bool): Whether to enable bounded ReLU clipping at output.
            bounds_low (torch.Tensor): Lower bounds (output_dim,).
            bounds_high (torch.Tensor): Upper bounds (output_dim,).
            mask (torch.Tensor): Binary tensor mask (output_dim,) for enabling output clamping.
        """
        super().__init__()
        # — default hidden_dim to 4×input_dim if not overridden —
        if hidden_dim is None:
            hidden_dim = 4 * input_dim

        self.columns = nn.ModuleList()
        self.adapters = []  # List[Tuple[nn.ModuleList, nn.ModuleList]]
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dims: List[int] = []  # Store per-task output dims

        # store bounds configuration
        self.use_bounds = use_bounds
        self.bounds_low = bounds_low
        self.bounds_high = bounds_high
        self.mask = mask

    def add_column(self, output_dim: int) -> None:
        """
        Add a new task-specific column with its own output dimension and lateral adapters.
        Also freezes all previous columns' parameters.
        """
        column_index = len(self.columns)
        logger.debug("Adding column %d", column_index)

        self.output_dims.append(output_dim)
        device = next(self.columns[0].parameters()).device if self.columns else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        output_dim=len(self.bounds_low)
        new_column = ProgressiveColumn(
            self.input_dim, self.hidden_dim, output_dim,
            use_bounds=self.use_bounds,
            bounds_low=self.bounds_low,
            bounds_high=self.bounds_high,
            mask=self.mask
        ).to(device)

        self.columns.append(new_column)

        for col in self.columns[:-1]:
            for param in col.parameters():
                param.requires_grad = False
            logger.debug("Froze parameters of a previous column")

        if column_index == 0:
            self.adapters.append((None, None))
        else:
            adapter1 = nn.ModuleList([
                AlphaAdapter(self.hidden_dim, reduction=4).to(device)
                for _ in range(column_index)
            ])
            adapter2 = nn.ModuleList([
                AlphaAdapter(self.hidden_dim, reduction=4).to(device)
                for _ in range(column_index)
            ])
            self.adapters.append((adapter1, adapter2))

            for param in itertools.chain.from_iterable(
                list(a1.parameters()) + list(a2.parameters())
                for (a1, a2) in self.adapters[:-1]
                if a1 is not None):
                param.requires_grad = False

    def forward(self, x: torch.Tensor, task_id: int) -> torch.Tensor:
        """
        Forward pass for a specific task column, integrating lateral inputs from previous columns.

        Args:
            x (torch.Tensor): Input tensor.
            task_id (int): Index of the column/task to use.

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, output_dim).
        """
        if task_id == 0:
            return self.columns[0](x)

        # Get lateral outputs from each previous column
        h1_list, h2_list = [], []
        for j in range(task_id):
            h1_j, h2_j, _ = self.columns[j].forward_with_activations(x)
            h1_list.append(h1_j)
            h2_list.append(h2_j)

        # Apply adapters
        h1_lateral = sum(self.adapters[task_id][0][j](h1_list[j]) for j in range(task_id))
        h2_lateral = sum(self.adapters[task_id][1][j](h2_list[j]) for j in range(task_id))

        # Forward through the new column with lateral input added
        h1 = self.columns[task_id].act1(self.columns[task_id].fc1(x) + h1_lateral)
        h2 = self.columns[task_id].act2(self.columns[task_id].fc2(h1) + h2_lateral)
        out = self.columns[task_id].output_layer(h2)
        return self.columns[task_id].bound_layer(out)

    def get_all_shared_weights(self) -> List[torch.nn.Parameter]:
        """
        Collect all trainable parameters from all columns and adapters.

        Returns:
            List of trainable weight tensors.
        """
        weights = []
        for col in self.columns:
            weights.extend(p for p in col.parameters() if p.requires_grad)
        for pair in self.adapters:
            if pair[0] is not None:
                weights.extend(p for p in pair[0].parameters() if p.requires_grad)
            if pair[1] is not None:
                weights.extend(p for p in pair[1].parameters() if p.requires_grad)
        return weights

    def prune_adapters(self, threshold: float = 1e-3) -> None:
        """
        Prune adapter weights across all tasks by zeroing out weights with magnitude below `threshold`.

        Args:
            threshold (float): Magnitude threshold for pruning.
        """
        for task_id, (adapters_h1, adapters_h2) in enumerate(self.adapters):
            if adapters_h1 is None:
                continue
            for j, adapter in enumerate(adapters_h1 + adapters_h2):
                for name, param in adapter.adapter.named_parameters():
                    mask = param.abs() >= threshold
                    num_pruned = (~mask).sum().item()
                    param.data *= mask
                    logger.debug(f"Pruned {num_pruned} weights in adapter[{task_id}][{j}].{name}")

    def prune_small_alpha_adapters(self, alpha_threshold=1e-3):
        for task_adapters in self.adapters:
            for layer_adapters in task_adapters:
                if layer_adapters is None:
                    continue
                for adapter in layer_adapters:
                    if adapter.alpha.abs().item() < alpha_threshold:
                        for param in adapter.parameters():
                            param.data.zero_()

    def log_adapter_sparsity(self) -> None:
        """
        Log the percentage of remaining (non-zero) weights in adapters.
        """
        for task_id, (a1, a2) in enumerate(self.adapters):
            if a1 is None:
                continue
            total = 0
            kept = 0
            for adapter in a1 + a2:
                for param in adapter.adapter.parameters():
                    total += param.numel()
                    kept += (param != 0).sum().item()
            logger.info(f"Task {task_id}: {kept}/{total} weights kept ({100 * kept / total:.2f}%)")

    def compute_afs(self, task_id: int, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, List[float]]:
        """
        Compute Average Fisher Sensitivity (AFS) for the lateral inputs into task_id column.

        Returns:
            Dict mapping 'layer1' and 'layer2' to lists of AFS scores (one per source column).
        """
        self.eval()
        afs_scores = {"layer1": [0.0 for _ in range(task_id)],
                      "layer2": [0.0 for _ in range(task_id)]}
        count = 0

        loss_fn = nn.MSELoss()

        for xb, yb,_ in dataloader:
            xb, yb = xb.to(device), yb.to(device)

            h1_inputs = []
            h2_inputs = []
            for j in range(task_id):
                h1_j, h2_j, _ = self.columns[j].forward_with_activations(xb)
                h1_j.retain_grad()
                h2_j.retain_grad()
                h1_inputs.append(h1_j)
                h2_inputs.append(h2_j)

            h1_lateral = sum(self.adapters[task_id][0][j](h1_inputs[j]) for j in range(task_id))
            h2_lateral = sum(self.adapters[task_id][1][j](h2_inputs[j]) for j in range(task_id))

            h1 = self.columns[task_id].act1(self.columns[task_id].fc1(xb) + h1_lateral)
            h2 = self.columns[task_id].act2(self.columns[task_id].fc2(h1) + h2_lateral)
            output = self.columns[task_id].output_layer(h2)
            loss = loss_fn(self.columns[task_id].bound_layer(output), yb)

            self.zero_grad()
            loss.backward()

            for j in range(task_id):
                afs_scores["layer1"][j] += h1_inputs[j].grad.norm(2, dim=1).pow(2).mean().item()
                afs_scores["layer2"][j] += h2_inputs[j].grad.norm(2, dim=1).pow(2).mean().item()
            count += 1

        for j in range(task_id):
            afs_scores["layer1"][j] /= count
            afs_scores["layer2"][j] /= count

        return afs_scores

    def compute_aps(self, task_id: int, dataloader: torch.utils.data.DataLoader, device: torch.device) -> Dict[str, List[float]]:
        """
        Compute APS = 1 / critical_sigma, where critical_sigma causes ~50% drop in performance.

        Returns dict of APS values for each lateral adapter per layer.
        """
        self.eval()
        aps_scores = {"layer1": [], "layer2": []}
        base_acc = self.evaluate_model(task_id, dataloader, device)

        for layer_idx, layer_name in enumerate(["layer1", "layer2"]):
            for j in range(task_id):  # previous columns
                sigma_crit = self.find_critical_sigma(task_id, layer_idx, j, dataloader, device, base_acc)
                aps = 1.0 / sigma_crit if sigma_crit > 0 else float("inf")
                aps_scores[layer_name].append(aps)

        return aps_scores

    def identify_dead_neurons(self, column_id, dataloader, device, eps=1e-5):
        """
        Identify neurons in a frozen column whose activations are always below epsilon.
        """
        self.eval()
        dead_neurons = {1: set(), 2: set()}
        activations1, activations2 = [], []

        for xb, _,_ in dataloader:
            xb = xb.to(device)
            h1, h2, _ = self.columns[column_id].forward_with_activations(xb)
            activations1.append(h1.detach().abs())
            activations2.append(h2.detach().abs())

        max1 = torch.cat(activations1).max(dim=0).values.to(device, non_blocking=True)
        max2 = torch.cat(activations2).max(dim=0).values.to(device, non_blocking=True)

        dead_neurons[1] = set((max1 < eps).nonzero(as_tuple=True)[0].tolist())
        dead_neurons[2] = set((max2 < eps).nonzero(as_tuple=True)[0].tolist())
        return dead_neurons

    def find_critical_sigma(self, task_id, layer_idx, column_j, dataloader, device, base_score, tolerance=0.05) -> float:
        """
        Binary search to find sigma where accuracy drops to ~50% of base_score.
        Returns the smallest such sigma.
        """
        low, high = 1e-4, 1.0
        for _ in range(10):
            mid_sigma = (low + high) / 2
            acc = self.evaluate_with_noise(task_id, layer_idx, column_j, dataloader, device, mid_sigma)
            if acc < 0.5 * base_score:
                high = mid_sigma
            else:
                low = mid_sigma
            if abs(acc - 0.5 * base_score) < tolerance * base_score:
                break
        return mid_sigma

    def evaluate_model(self, task_id: int, dataloader: torch.utils.data.DataLoader, device: torch.device) -> float:
        """
        Evaluate model performance on a regression task using Mean Squared Error (MSE).

        Args:
            task_id (int): Column index corresponding to the current task.
            dataloader (DataLoader): Dataloader for evaluation data.
            device (torch.device): Device to run evaluation on.

        Returns:
            float: Mean squared error over all samples.
        """
        self.eval()
        total_loss = 0.0
        count = 0

        for xb, yb,_ in dataloader:
            xb, yb = xb.to(device), yb.to(device)
            preds = self.forward(xb, task_id)
            total_loss += F.mse_loss(preds, yb, reduction="sum").item()
            count += yb.numel()  # total number of scalar elements

        return total_loss / count if count > 0 else 0.0

    def evaluate_with_noise(
        self,
        task_id: int,
        layer_idx: int,
        column_j: int,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        sigma: float
    ) -> float:
        """
        Inject Gaussian noise into a specific lateral adapter input and evaluate regression performance
        using Mean Squared Error (MSE).

        Args:
            task_id (int): Task index to evaluate.
            layer_idx (int): Adapter layer index (0 or 1).
            column_j (int): Source column index to inject noise into.
            dataloader (DataLoader): Evaluation data loader.
            device (torch.device): Device to run evaluation on.
            sigma (float): Standard deviation of Gaussian noise to inject.

        Returns:
            float: Average mean squared error over the dataset.
        """
        self.eval()
        total_loss = 0.0
        count = 0

        for xb, yb in dataloader:
            xb, yb = xb.to(device), yb.to(device)

            h1_j, h2_j, _ = self.columns[column_j].forward_with_activations(xb)
            features = h1_j if layer_idx == 0 else h2_j

            noisy_input = features + sigma * torch.randn_like(features)
            adapted = self.adapters[task_id][layer_idx][column_j](noisy_input)
            sum_other = sum(
                self.adapters[task_id][layer_idx][j](h1_j if layer_idx == 0 else h2_j)
                for j in range(task_id) if j != column_j
            )
            total_lat = adapted + sum_other

            x_fc1 = self.columns[task_id].fc1(xb)
            x_fc1 = x_fc1 + total_lat if layer_idx == 0 else x_fc1
            x_act1 = F.relu(x_fc1)
            x_fc2 = self.columns[task_id].fc2(x_act1)
            x_act2 = F.relu(x_fc2)
            out = self.columns[task_id].output_layer(x_act2)
            out = self.columns[task_id].bound_layer(out)

            total_loss += F.mse_loss(out, yb, reduction="sum").item()
            count += yb.numel()

        return total_loss / count if count > 0 else 0.0
