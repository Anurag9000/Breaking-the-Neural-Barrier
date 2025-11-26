# import torch

# def soft_threshold_l1(params, lam: float):
#     """
#     Performs element‐wise L1 soft‐thresholding (proximal operator) on all tensors in `params`.
#     Equivalent to: p = sign(p) * max(|p| - lam, 0)
#     """
#     with torch.no_grad():
#         for p in params:
#             # only threshold leaf tensors
#             if p.requires_grad:
#                 p.data = torch.where(
#                     p.data.abs() < lam,
#                     torch.zeros_like(p.data),
#                     p.data - lam * p.data.sign()
#                 )

# def group_l2_sparsify(weight: torch.Tensor, gl_lambda: float) -> torch.Tensor:
#     """
#     Performs group‐L2 shrinkage on the columns of `weight`.
#     For each column j: scale = max(1 - gl_lambda / ||col_j||₂, 0)
#     Returns the sparsified weight.
#     """
#     # compute ℓ₂ norm per column
#     norms = weight.norm(p=2, dim=0, keepdim=True)  # shape [1, out_features]
#     # avoid division by zero
#     scale = torch.clamp(1.0 - gl_lambda / (norms + 1e-12), min=0.0)
#     return weight * scale
