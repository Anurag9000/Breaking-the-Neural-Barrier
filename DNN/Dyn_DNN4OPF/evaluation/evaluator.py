"""
evaluator.py
============

Purpose:
    Contains evaluation metrics and interpretability tools for regression models on OPF data.

Functionality:
    - Implements MSE, MAE, RMSE, R², MAPE, MaxError
    - `Evaluator` class supports:
        - Feature ablation
        - Gradient-based feature importance
        - Saliency maps
        - Interpretability summary with named features

Paper Role:
    Allows quantitative comparison of forgetting/generalization in continual learning models.
    Also supports AFS/APS (Average Fisher Sensitivity and Perturbation Sensitivity) for PNN analysis.
"""

import logging
from typing import Callable, Dict, List, Optional, Union
import torch
import numpy as np

# Set up logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Adjust level as needed


def mse(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    reduction: str = 'mean'
) -> Union[float, torch.Tensor]:
    """
    Mean Squared Error (MSE) between predictions and targets.

    Args:
        y_pred (torch.Tensor): Predicted values.
        y_true (torch.Tensor): True values.
        reduction (str): 'mean' returns scalar MSE, 'none' returns element-wise error.

    Returns:
        float or torch.Tensor: MSE loss (scalar or tensor depending on reduction).
    """
    err = (y_pred - y_true) ** 2
    if reduction == 'mean':
        result = err.mean().item()
        logger.debug(f"MSE (mean): {result}")
        return result
    elif reduction == 'none':
        logger.debug("MSE (none): returning element-wise error tensor")
        return err
    else:
        logger.error(f"Invalid reduction parameter: {reduction}")
        raise ValueError(reduction)


def mae(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    reduction: str = 'mean'
) -> Union[float, torch.Tensor]:
    """
    Mean Absolute Error (MAE) between predictions and targets.

    Args and Returns similar to mse().
    """
    err = (y_pred - y_true).abs()
    if reduction == 'mean':
        result = err.mean().item()
        logger.debug(f"MAE (mean): {result}")
        return result
    elif reduction == 'none':
        logger.debug("MAE (none): returning element-wise error tensor")
        return err
    else:
        logger.error(f"Invalid reduction parameter: {reduction}")
        raise ValueError(reduction)


def rmse(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    reduction: str = 'mean'  # included param for consistency though unused
) -> float:
    """
    Root Mean Squared Error between predictions and targets.

    Returns:
        float: RMSE scalar.
    """
    rmse_value = torch.sqrt(((y_pred - y_true) ** 2).mean()).item()
    logger.debug(f"RMSE: {rmse_value}")
    return rmse_value


def mape(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor, 
    reduction: str = 'mean', 
    eps: float = 1e-7
) -> Union[float, torch.Tensor]:
    """
    Mean Absolute Percentage Error (MAPE) between predictions and targets.

    Args:
        eps: Small constant to avoid division by zero.

    Returns:
        float or torch.Tensor: MAPE in percentage.
    """
    err = ((y_pred - y_true).abs() / (y_true.abs() + eps))
    if reduction == 'mean':
        result = (err * 100).mean().item()
        logger.debug(f"MAPE (mean): {result}")
        return result
    elif reduction == 'none':
        logger.debug("MAPE (none): returning element-wise percentage error tensor")
        return err * 100
    else:
        logger.error(f"Invalid reduction parameter: {reduction}")
        raise ValueError(reduction)


def r2_score(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor
) -> float:
    """
    Coefficient of determination (R²) score.

    Returns:
        float: Average R² over outputs or scalar if single output.
    """
    y_true_mean = y_true.mean(dim=0)
    ss_tot = ((y_true - y_true_mean) ** 2).sum(dim=0)
    ss_res = ((y_true - y_pred) ** 2).sum(dim=0)
    r2 = 1 - ss_res / (ss_tot + 1e-8)  # Avoid divide by zero
    result = r2.mean().item() if r2.ndim else r2.item()
    logger.debug(f"R2 score: {result}")
    return result


def max_error(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor
) -> float:
    """
    Maximum absolute error between predictions and targets.

    Returns:
        float: Maximum error.
    """
    err = (y_pred - y_true).abs()
    max_err = err.max().item()
    logger.debug(f"Max error: {max_err}")
    return max_err


def per_output_mse(y_pred: torch.Tensor, y_true: torch.Tensor) -> Dict[str, float]:
    """
    Compute MSE for each output dimension.
    Returns a dict mapping 'output_{i}_mse' to the MSE value (as a float).
    """
    mse_dict = {
        f"output_{i}_mse": ((y_pred[:, i] - y_true[:, i]) ** 2).mean().item()
        for i in range(y_pred.shape[1])
    }
    logger = logging.getLogger(__name__)
    # logger.debug("per_output_mse → " +
    #              ", ".join(f"{k}={v:.6f}" for k, v in mse_dict.items()))
    return mse_dict


def per_output_metrics(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor
) -> Dict[str, float]:
    """
    Calculate metrics for each output dimension independently (multi-output regression).

    Returns:
        dict: Metric names mapped to their values per output dimension.
    """
    metrics = {}
    for i in range(y_pred.shape[1]):
        metrics[f'output_{i}_mse'] = mse(y_pred[:, i], y_true[:, i])
        metrics[f'output_{i}_mae'] = mae(y_pred[:, i], y_true[:, i])
        metrics[f'output_{i}_rmse'] = rmse(y_pred[:, i], y_true[:, i])
        metrics[f'output_{i}_mape'] = mape(y_pred[:, i], y_true[:, i])
        metrics[f'output_{i}_r2'] = r2_score(y_pred[:, i], y_true[:, i])
        metrics[f'output_{i}_max_error'] = max_error(y_pred[:, i], y_true[:, i])
    logger.debug("Per-output metrics computed.")
    return metrics


def all_metrics(
    y_pred: torch.Tensor, 
    y_true: torch.Tensor
) -> Dict[str, float]:
    """
    Compute all main regression metrics on given predictions and targets.

    Returns:
        dict: Metric names mapped to their scalar values.
    """
    metrics_dict = {
        'MSE': mse(y_pred, y_true),
        'MAE': mae(y_pred, y_true),
        'RMSE': rmse(y_pred, y_true),
        'MAPE': mape(y_pred, y_true),
        'R2': r2_score(y_pred, y_true),
        'MaxError': max_error(y_pred, y_true)
    }
    logger.debug("All metrics computed.")
    return metrics_dict


class Evaluator:
    """
    Class to evaluate regression models and perform interpretability analyses
    such as ablation studies, feature importance, and saliency mapping.
    """

    def __init__(self, model: torch.nn.Module, device: Optional[torch.device] = None) -> None:
        """
        Initialize Evaluator.

        Args:
            model (torch.nn.Module): Model to evaluate.
            device (torch.device, optional): Device to run evaluation on.
                Defaults to CUDA if available, else CPU.
        """
        self.model = model
        self.model.eval()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        logger.info(f"Evaluator initialized on device: {self.device}")

    def ablation_study(
        self, 
        data_loader: torch.utils.data.DataLoader, 
        ablate_type: str = "feature", 
        ablate_indices: Optional[List[int]] = None,
        metric_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None
    ) -> Dict[int, float]:
        """
        Perform an ablation study to measure impact of ablating features or layers.

        Args:
            data_loader: DataLoader providing (inputs, targets).
            ablate_type: Either "feature" to zero-out input features or "layer" to zero activations.
            ablate_indices: List of feature or layer indices to ablate.
            metric_fn: Function computing a metric from (predictions, targets).

        Returns:
            Dict mapping ablated index to metric degradation.
        """
        if ablate_indices is None:
            logger.warning("ablate_indices is None; returning empty results.")
            return {}

        results = {}
        logger.info(f"Starting ablation study: type={ablate_type}, indices={ablate_indices}")
        if ablate_type == "feature":
            for idx in ablate_indices:
                degradation = self._ablate_feature(data_loader, idx, metric_fn)
                results[idx] = degradation
                logger.debug(f"Feature {idx} ablation degradation: {degradation}")
        elif ablate_type == "layer":
            for idx in ablate_indices:
                degradation = self._ablate_layer(data_loader, idx, metric_fn)
                results[idx] = degradation
                logger.debug(f"Layer {idx} ablation degradation: {degradation}")
        else:
            logger.error(f"Unknown ablate_type: {ablate_type}")
        return results

    def _ablate_feature(
        self, 
        data_loader: torch.utils.data.DataLoader, 
        feature_idx: int,
        metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> float:
        """
        Ablate (zero) a single feature and measure metric change.

        Returns metric difference: ablated - normal
        """
        metric_normal = self._compute_metric(data_loader, metric_fn)
        metric_ablate = self._compute_metric(data_loader, metric_fn, ablate_feature=feature_idx)
        return metric_ablate - metric_normal

    def _ablate_layer(
        self, 
        data_loader: torch.utils.data.DataLoader, 
        layer_idx: int, 
        metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
    ) -> float:
        """
        Ablate (zero) activations of a given layer using a forward hook.

        Returns metric after ablation.
        """
        handles = []
        # Collect all Linear layers
        layers = [m for m in self.model.modules() if isinstance(m, torch.nn.Linear)]

        def hook_fn(module, input, output):
            return torch.zeros_like(output)

        # Register hook on specified layer
        handles.append(layers[layer_idx].register_forward_hook(hook_fn))
        metric_ablate = self._compute_metric(data_loader, metric_fn)
        # Remove hook to restore normal behavior
        for h in handles:
            h.remove()
        return metric_ablate

    def _compute_metric(
        self, 
        data_loader: torch.utils.data.DataLoader, 
        metric_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
        ablate_feature: Optional[int] = None
    ) -> float:
        """
        Compute average metric on data loader, optionally ablating a feature by zeroing.

        Args:
            ablate_feature: Index of input feature to zero out.

        Returns:
            Average metric value (float).
        """
        self.model.eval()
        scores = []
        for xb, yb in data_loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            if ablate_feature is not None:
                xb = xb.clone()
                xb[:, ablate_feature] = 0
            pred = self.model(xb)
            score = metric_fn(pred, yb)
            scores.append(score.item())
        avg_score = np.mean(scores)
        logger.debug(f"Computed metric (ablate_feature={ablate_feature}): {avg_score}")
        return avg_score

    def feature_importance(
        self, 
        data_loader: torch.utils.data.DataLoader, 
        method: str = "gradient", 
        top_k: int = 10
    ) -> np.ndarray:
        """
        Estimate feature importance using specified method.

        Args:
            method: "gradient" for saliency, "input_perturb" for brute-force ablation.
            top_k: Number of top features to return (currently unused).

        Returns:
            Numpy array of importance scores per feature.
        """
        logger.info(f"Computing feature importance using method: {method}")
        if method == "gradient":
            return self._feature_importance_gradient(data_loader)
        elif method == "input_perturb":
            return self._feature_importance_perturb(data_loader)
        else:
            logger.error(f"Feature importance method not implemented: {method}")
            raise NotImplementedError(method)

    def _feature_importance_gradient(
        self, 
        data_loader: torch.utils.data.DataLoader
    ) -> np.ndarray:
        """
        Compute gradient-based feature importance (saliency) averaged over data.

        Returns:
            Numpy array with mean absolute gradients per input feature.
        """
        self.model.eval()
        grads = []
        for xb, yb in data_loader:
            xb = xb.to(self.device).detach().requires_grad_(True)
            yb = yb.to(self.device)
            out = self.model(xb)
            loss = torch.nn.functional.mse_loss(out, yb)
            loss.backward()
            grads.append(xb.grad.abs().mean(dim=0).detach().cpu().numpy())
        mean_grads = np.mean(grads, axis=0)
        logger.debug("Gradient-based feature importance computed.")
        return mean_grads

    def _feature_importance_perturb(
        self, 
        data_loader: torch.utils.data.DataLoader
    ) -> np.ndarray:
        """
        Compute feature importance by ablating each feature individually
        and measuring metric drop (slow brute-force method).

        Returns:
            Numpy array of importance scores per feature.
        """
        input_dim = next(iter(data_loader))[0].shape[1]
        metric_fn = lambda pred, y: torch.nn.functional.mse_loss(pred, y)
        base_score = self._compute_metric(data_loader, metric_fn)
        importances = []
        for i in range(input_dim):
            ablated_score = self._compute_metric(data_loader, metric_fn, ablate_feature=i)
            importances.append(ablated_score - base_score)
        logger.debug("Input perturbation-based feature importance computed.")
        return np.array(importances)

    def saliency_map(
        self, 
        xb: torch.Tensor, 
        method: str = "gradient"
    ) -> np.ndarray:
        """
        Compute saliency map for a single input sample.

        Args:
            xb: Input tensor of shape (1, input_dim).
            method: Only 'gradient' supported currently.

        Returns:
            1D numpy array representing input feature saliency.
        """
        xb = xb.to(self.device).detach().requires_grad_(True)
        self.model.zero_grad()
        output = self.model(xb)
        output = output.sum()
        output.backward()
        saliency = xb.grad.abs().squeeze().detach().cpu().numpy()
        logger.debug("Saliency map computed for one input sample.")
        return saliency

    def interpret(
        self, 
        data_loader: torch.utils.data.DataLoader, 
        methods: Optional[List[str]] = None, 
        feature_names: Optional[List[str]] = None
    ) -> Dict[str, Union[Dict[str, float], np.ndarray]]:
        """
        Run interpretability analyses: ablation, feature importance, and saliency.

        Args:
            methods: List of methods to run; defaults to all.
            feature_names: Optional list to label feature-related results.

        Returns:
            Dictionary of interpretability results.
        """
        if methods is None:
            methods = ["ablation", "feature_importance", "saliency"]
        results = {}

        if "ablation" in methods:
            results["ablation"] = self.ablation_study(
                data_loader,
                ablate_type="feature",
                ablate_indices=list(range(next(iter(data_loader))[0].shape[1])),
                metric_fn=lambda pred, y: torch.nn.functional.mse_loss(pred, y)
            )

        if "feature_importance" in methods:
            results["feature_importance"] = self.feature_importance(data_loader, method="gradient")

        if "saliency" in methods:
            xb, _ = next(iter(data_loader))
            xb = xb[0].unsqueeze(0)  # Take first sample only
            results["saliency"] = self.saliency_map(xb)

        # If feature names provided, label the results
        if feature_names is not None:
            for key in ["ablation", "feature_importance", "saliency"]:
                if key in results:
                    if isinstance(results[key], dict):
                        results[key] = dict(zip(feature_names, results[key].values()))
                    elif isinstance(results[key], np.ndarray):
                        results[key] = dict(zip(feature_names, results[key]))

        logger.info("Interpretability pipeline completed.")
        return results

# USAGE:
# evaluator = Evaluator(model)
# ablation_results = evaluator.ablation_study(loader, ablate_type="feature", ablate_indices=[0,1,2], metric_fn=mse)
# fi = evaluator.feature_importance(loader)
# saliency = evaluator.saliency_map(xb)
# results = evaluator.interpret(loader, feature_names=[...])

# Example usage:
# if __name__ == "__main__":
#     # Simulate some predictions and targets (batch_size x output_dim)
#     y_pred = torch.tensor([[2.5, 0.5], [0.0, 2.0], [2.1, 1.3]])
#     y_true = torch.tensor([[3.0, -0.5], [0.0, 2.0], [2.0, 1.0]])
#     print("All metrics:", all_metrics(y_pred
