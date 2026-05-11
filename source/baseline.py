"""
Mode B baseline: shared MLP encoder + per-instance closed-form linear head.

Architecture
============
    state features (n_state)            instance embedding (n_inst, from policy encoder)
              │                                          │
              └──────────────────────┬───────────────────┘
                                     │
                              ┌──────▼──────┐
                              │  Linear     │
                              │  ReLU       │   ← shared θ across all instances
                              │  Linear     │      trained jointly via MSE
                              └──────┬──────┘
                                     │
                              φ ∈ R^h (learned features)
                                     │
                              ⟨φ, β_b⟩           ← per-instance linear head,
                                     │             closed-form OLS each batch
                                     ▼
                                 H_hat
"""

import torch
import torch.nn as nn


class EntropyBaselineMLP(nn.Module):
    """MLP that produces h-dim features for per-instance OLS."""

    def __init__(self, n_state, n_inst, hidden, h_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_state + n_inst, hidden),
            nn.ReLU(),
            nn.Linear(hidden, h_out),
        )

    def forward(self, state_feat, inst_emb_per_step):
        """
        state_feat:        (B, P, T, n_state)
        inst_emb_per_step: (B, P, T, n_inst)
        Returns features:  (B, P, T, h_out)
        """
        return self.net(torch.cat([state_feat, inst_emb_per_step], dim=-1))


def batched_per_instance_ols(features, H, valid_mask, ridge=1e-4):
    """
    For each instance b ∈ {1..B}, solve closed-form ridge OLS:
        β_b = argmin || H_b - φ_b · β ||²  +  ridge · ||β||²
    using only valid steps.

    features:   (B, P, T, h)
    H:          (B, P, T)
    valid_mask: (B, P, T) bool — True means include in fit
    ridge:      scalar — Tikhonov regularization

    Returns:
        H_hat: (B, P, T)        predicted entropy
        beta:  (B, h)           per-instance coefficient vector
    """
    B, P, T, h = features.shape
    X = features.reshape(B, P * T, h)                       # (B, n, h)
    y = H.reshape(B, P * T, 1)                              # (B, n, 1)
    m = valid_mask.reshape(B, P * T, 1).float()             # (B, n, 1)
    Xm = X * m
    ym = y * m
    XtX = Xm.transpose(1, 2) @ Xm                           # (B, h, h)
    XtX = XtX + ridge * torch.eye(h, device=X.device).unsqueeze(0)
    Xty = Xm.transpose(1, 2) @ ym                           # (B, h, 1)
    beta = torch.linalg.solve(XtX, Xty).squeeze(-1)         # (B, h)
    H_hat = (X @ beta.unsqueeze(-1)).squeeze(-1).reshape(B, P, T)
    return H_hat, beta


def extract_state_features_tsp(state, problem_size):
    """
    TSP per-step state features (4 dims) — derived from `state` returned by env.
    All tensors are (B, P).

    Returns: (B, P, 4) float tensor
    """
    nf = state.n_feasible.float()
    sc = float(state.selected_count)
    sc_norm = torch.full_like(nf, sc / max(problem_size, 1))

    return torch.stack([
        torch.log(nf.clamp(min=1.0)),       # log F
        nf / problem_size,                   # mask fraction
        sc_norm,                             # progress
        sc_norm.square(),                    # quadratic progress
    ], dim=-1)                               # (B, P, 4)


def extract_state_features_cvrp(state, problem_size):
    """
    CVRP per-step state features (8 dims).
    All tensors are (B, P) at the time of the model decision.

    At selected_count == 0 (env.pre_step), state.current_node is None — we
    return safe zero indicators for the depot-related features. These steps
    are masked out of the OLS fit anyway (forced_steps=2).

    Returns: (B, P, 8) float tensor
    """
    nf = state.n_feasible.float()
    load = state.load
    if state.current_node is None:
        at_depot = torch.zeros_like(nf)
    else:
        at_depot = (state.current_node == 0).float()
    sc = float(state.selected_count)
    sc_norm = torch.full_like(nf, sc / max(2 * problem_size, 1))  # ≤ 1 in normal CVRP

    return torch.stack([
        torch.log(nf.clamp(min=1.0)),                 # log F
        nf / (problem_size + 1),                       # mask fraction
        load,                                          # capacity remaining
        at_depot,                                      # depot step indicator
        torch.log(load.clamp(min=1e-3)),               # log load (different scale)
        sc_norm,                                       # progress
        1.0 - load,                                    # demand consumed
        at_depot * (1.0 - sc_norm),                    # early-depot interaction
    ], dim=-1)                                         # (B, P, 8)


def extract_state_features_vrptw(state, problem_size):
    """
    VRPTW state features. Uses current_time instead of load.

    Same None-guard as CVRP for current_node at selected_count == 0.
    """
    nf = state.n_feasible.float()
    if state.current_node is None:
        at_depot = torch.zeros_like(nf)
    else:
        at_depot = (state.current_node == 0).float()
    sc = float(state.selected_count)
    sc_norm = torch.full_like(nf, sc / max(2 * problem_size, 1))
    ct = state.current_time   # normalized in [0, ~1] typically

    return torch.stack([
        torch.log(nf.clamp(min=1.0)),
        nf / (problem_size + 1),
        ct,
        at_depot,
        sc_norm,
        sc_norm.square(),
        at_depot * (1.0 - sc_norm),
        ct * (1.0 - at_depot),
    ], dim=-1)                                         # (B, P, 8)


# ---------------------------------------------------------------------------
# weights computation (Mode B): uses residual = H - H_hat from MLP+OLS,
# then performs the same group-by-F z-score → softmax × T_valid as the original
# entropy_utils.compute_entropy_weights.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_mode_b_weights(residual, n_feasible, advantage, valid_mask, gamma):
    """
    residual:   (B, P, T) — H - H_hat (already computed)
    n_feasible: (B, P, T) — for group-by-F z-score
    advantage:  (B, P)    — trajectory-level
    valid_mask: (B, P, T) bool — True = real decision step
    gamma:      float

    Returns:
        weights: (B, P, T) — sum over T per (B,P) ≈ T_valid
                  invalid steps get weight ≈ 0 (their log p = 0 anyway)
    """
    B, P, T = residual.shape

    # ── Group-by-F z-score, using only valid steps ────────────────────────────
    nf = n_feasible.reshape(B, -1).long()
    res_flat = residual.reshape(-1)
    v_flat = valid_mask.reshape(-1).float()

    max_nf = nf.max() + 1
    bid = torch.arange(B, device=residual.device)[:, None].expand_as(nf)
    gid = (bid * max_nf + nf).reshape(-1)
    n_grp = B * max_nf

    # weighted (by validity) sums for group statistics
    counts = torch.zeros(n_grp, device=residual.device).scatter_add_(0, gid, v_flat)
    sums   = torch.zeros(n_grp, device=residual.device).scatter_add_(0, gid, res_flat * v_flat)
    grp_mean = (sums / counts.clamp(min=1))[gid]

    centered = (res_flat - grp_mean) * v_flat
    sq_sums  = torch.zeros(n_grp, device=residual.device).scatter_add_(0, gid, centered.square())
    grp_std  = ((sq_sums / counts.clamp(min=1)).sqrt().clamp(min=1e-8))[gid]

    delta_H = ((res_flat - grp_mean) / grp_std).reshape(B, P, T)

    # ── Sign by advantage → softmax over T → × T_valid ────────────────────────
    sign_A = advantage.sign().unsqueeze(2)
    logit  = gamma * sign_A * delta_H
    logit  = logit.masked_fill(~valid_mask, -1e9)

    sm = torch.softmax(logit, dim=2)
    T_valid = valid_mask.sum(dim=2, keepdim=True).float().clamp(min=1.0)
    w = sm * T_valid

    return w
