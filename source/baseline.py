"""
Mode B baseline: shared MLP encoder + per-instance closed-form linear head.

Input philosophy (MLP mode)
===========================
We feed the baseline ONLY policy-encoder outputs (instance environment
representation) and raw env scalars. No hand-engineered nonlinearities
(no log / ratio / square / interaction) — the MLP learns those itself.
No decoder-side tensors (no `q`, no `mh_out`) — those carry policy
selection-confidence and would absorb the very signal we want as residual.

Concretely, the caller (Train.py) builds:
    TSP : [inst_summary, enc_last, enc_first, F, sc, current_xy]      (3D + 4)
    CVRP: [inst_summary, enc_last,            F, sc, load, current_xy, vis_count]  (2D + 6)
and feeds the whole vector to EntropyBaselineMLP.forward(x).

Architecture
============
    raw input  ── Linear ─ ReLU ─ Linear ─ ReLU ─ Linear ──►  φ ∈ R^h_out
                                                                │
                              ⟨φ, β_b⟩    ← per-instance OLS    │
                                                                ▼
                                                              H_hat
"""

import torch
import torch.nn as nn


class EntropyBaselineMLP(nn.Module):
    """MLP that produces h-dim features for per-instance OLS.

    Single-input interface: caller is responsible for concatenating
    (inst_summary, enc_last, enc_first?, raw_scalar) before calling forward.
    """

    def __init__(self, n_in, hidden=64, h_out=8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in,   hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, h_out),
        )

    def forward(self, x):
        """x: (..., n_in)  →  (..., h_out)."""
        return self.net(x)


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


# ---------------------------------------------------------------------------
# Hand-feature extractors — fed DIRECTLY into per-instance OLS
# (no MLP in between). Last column is always the intercept (constant 1).
# ---------------------------------------------------------------------------

def extract_hand_features_tsp(state, problem_size):
    """TSP hand features: [log F, 1]  (2 features)."""
    nf = state.n_feasible.float()
    log_F = torch.log(nf.clamp(min=1.0))
    return torch.stack([log_F, torch.ones_like(log_F)], dim=-1)   # (B, P, 2)


def extract_hand_features_cvrp(state, problem_size):
    """CVRP hand features: [log F, load, at_depot, visited_customer_ratio, 1]."""
    nf = state.n_feasible.float()
    load = state.load
    if state.current_node is None:
        at_depot = torch.zeros_like(nf)
    else:
        at_depot = (state.current_node == 0).float()
    vis_ratio = state.visited_customer_count / max(problem_size, 1)
    return torch.stack([
        torch.log(nf.clamp(min=1.0)),
        load,
        at_depot,
        vis_ratio,
        torch.ones_like(nf),
    ], dim=-1)                                                     # (B, P, 5)


def extract_hand_features_vrptw(state, problem_size):
    """VRPTW hand features: [log F, current_time, at_depot, visited_customer_ratio, 1]."""
    nf = state.n_feasible.float()
    if state.current_node is None:
        at_depot = torch.zeros_like(nf)
    else:
        at_depot = (state.current_node == 0).float()
    vis_ratio = state.visited_customer_count / max(problem_size, 1)
    return torch.stack([
        torch.log(nf.clamp(min=1.0)),
        state.current_time,
        at_depot,
        vis_ratio,
        torch.ones_like(nf),
    ], dim=-1)                                                     # (B, P, 5)


# ---------------------------------------------------------------------------
# Shared downstream reweighting: given a per-step residual (computed via either
# hand-feature OLS or MLP-feature OLS), do group-by-F z-score → softmax × T_valid.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_residual_weights(residual, n_feasible, advantage, valid_mask, gamma):
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
