"""
Hand-feature entropy weighting — per-instance multivariate OLS on hand-crafted
features (no MLP).

Pipeline (identical to the MLP-feature mode downstream, just different residual):
    1. per-instance OLS:  H_hat = X · β_b
    2. residual:          r = H − H_hat
    3. group-by-F z-score on residuals (excluding invalid steps)
    4. softmax over time, normalized by T_valid

Caller supplies the feature matrix X explicitly (one of the
`extract_hand_features_*` helpers in source.baseline).  TSP uses 2 features
[log F, 1]; CVRP / VRPTW use 5 features [log F, (load|time), at_depot,
visited_customer_ratio, 1].
"""

import torch

from source.baseline import batched_per_instance_ols, compute_residual_weights


@torch.no_grad()
def compute_entropy_weights(entropy, features, n_feasible, advantage,
                             valid_mask, gamma, ridge=1e-4):
    """
    Args:
        entropy:    (B, P, T)
        features:   (B, P, T, d) — OLS regressors (last column = intercept)
        n_feasible: (B, P, T)
        advantage:  (B, P)
        valid_mask: (B, P, T) bool — True = include in OLS + softmax
        gamma:      float — softmax temperature
        ridge:      float — Tikhonov regularization for OLS

    Returns:
        weights: (B, P, T), sum over T per (B,P) ≈ T_valid per trajectory
                 (invalid steps get weight ≈ 0)
    """
    H_hat, _ = batched_per_instance_ols(features, entropy, valid_mask, ridge=ridge)
    residual = entropy - H_hat
    return compute_residual_weights(residual, n_feasible, advantage, valid_mask, gamma=gamma)
