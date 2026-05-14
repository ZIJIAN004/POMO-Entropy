"""
Smoke test for the pure group-wise entropy reweighting path.

Covers:
  1. build_group_id shapes & ranges for TSP / CVRP
  2. compute_entropy_z_weights — invariants:
     (a) invalid steps      → c_t = 0
     (b) small-group steps  → c_t = 1 (no perturbation)
     (c) sufficient group   → mean(c|valid&sufficient) ≈ 1
  3. warmup mode (apply_perturbation=False) → all valid c_t == 1
  4. diagnostics: top3_concentration ∈ [0,1], small_group_ratio ∈ [0,1]

Run:
    python test_baseline_smoke.py
"""
import torch
import sys
sys.path.insert(0, '.')

from source.baseline import build_group_id, compute_entropy_z_weights

torch.manual_seed(0)

B, P, T = 4, 8, 30
N_BINS  = 10
MIN_GRP = 4


# === (1) build_group_id shapes ===============================================
n_feasible = torch.randint(1, 50, (B, P, T)).float()
gid_tsp, ngrp_tsp = build_group_id('tsp', n_feasible=n_feasible, n_bins=N_BINS)
assert gid_tsp.shape == (B, P, T) and gid_tsp.dtype == torch.long
assert gid_tsp.max().item() < ngrp_tsp
print(f"[1a] TSP gid: shape={tuple(gid_tsp.shape)}  n_grp_per_inst={ngrp_tsp}")

at_depot  = torch.randint(0, 2, (B, P, T)).float()
load      = torch.rand(B, P, T)
vis_ratio = torch.rand(B, P, T)
gid_cv, ngrp_cv = build_group_id(
    'cvrp', n_feasible=n_feasible, at_depot=at_depot,
    load=load, vis_ratio=vis_ratio, n_bins=N_BINS)
assert gid_cv.shape == (B, P, T) and gid_cv.dtype == torch.long
assert gid_cv.max().item() < ngrp_cv
print(f"[1b] CVRP gid: shape={tuple(gid_cv.shape)}  n_grp_per_inst={ngrp_cv}")


# === (2) compute_entropy_z_weights invariants ================================
# craft a controlled scenario:
#   - first 2 timesteps invalid (forced)
#   - make group=0 deliberately small (only a few steps belong)
#   - majority of steps belong to a large group → sufficient
entropy   = torch.randn(B, P, T) + 2.0
valid     = torch.ones(B, P, T, dtype=torch.bool)
valid[:, :, :2] = False
advantage = torch.randn(B, P)

w, diag = compute_entropy_z_weights(
    entropy=entropy, valid_mask=valid, advantage=advantage,
    gid=gid_tsp, n_grp_per_inst=ngrp_tsp,
    gamma=0.3, min_group_size=MIN_GRP, apply_perturbation=True)

# (a) invalid steps → w == 0
assert w[:, :, :2].abs().max().item() < 1e-7, "invalid steps must have w=0"
print(f"[2a] invalid steps w=0 ✓")

# (b) sufficient-group valid steps: mean ≈ 1 (c_t is linear perturbation around 1)
v_mean = w[valid].mean().item()
assert abs(v_mean - 1.0) < 0.3, f"valid w should center near 1, got {v_mean}"
print(f"[2b] mean(w|valid)={v_mean:.3f} (close to 1) ✓")

# (c) shape + finite
assert w.shape == (B, P, T)
assert torch.isfinite(w).all(), "weights must be finite"
print(f"[2c] shape={tuple(w.shape)}, all finite ✓")


# === (3) warmup mode: apply_perturbation=False → all valid w == 1 ============
w_wm, diag_wm = compute_entropy_z_weights(
    entropy=entropy, valid_mask=valid, advantage=advantage,
    gid=gid_tsp, n_grp_per_inst=ngrp_tsp,
    gamma=0.3, min_group_size=MIN_GRP, apply_perturbation=False)
assert (w_wm[valid] - 1.0).abs().max().item() < 1e-6, "warmup: valid w must all be exactly 1"
assert w_wm[~valid].abs().max().item() < 1e-7, "warmup: invalid w must be 0"
print(f"[3] warmup: valid w==1, invalid w==0 ✓")


# === (4) diagnostics in [0,1] ================================================
for k in ('top3_concentration', 'small_group_ratio'):
    v = diag[k].item()
    assert 0.0 <= v <= 1.0 + 1e-6, f"{k}={v} out of [0,1]"
    print(f"[4] {k}={v:.4f} ∈ [0,1] ✓")


# === (5) CVRP-shape end-to-end ===============================================
w_cv, diag_cv = compute_entropy_z_weights(
    entropy=entropy, valid_mask=valid, advantage=advantage,
    gid=gid_cv, n_grp_per_inst=ngrp_cv,
    gamma=0.3, min_group_size=MIN_GRP, apply_perturbation=True)
assert w_cv.shape == (B, P, T)
assert torch.isfinite(w_cv).all()
# CVRP has way more groups → expect many small groups (most ΔH set to 0)
v_mean_cv = w_cv[valid].mean().item()
assert abs(v_mean_cv - 1.0) < 0.3, f"CVRP valid w mean off: {v_mean_cv}"
print(f"[5] CVRP end-to-end: mean(w|valid)={v_mean_cv:.3f}, "
      f"small_group_ratio={diag_cv['small_group_ratio'].item():.3f} ✓")


# === (6) Bidir + softmax ablation switches ===================================
# (6a) bidir-only: sanity check — still finite, valid mean still near 1
w_bd, _ = compute_entropy_z_weights(
    entropy=entropy, valid_mask=valid, advantage=advantage,
    gid=gid_tsp, n_grp_per_inst=ngrp_tsp,
    gamma=0.3, min_group_size=MIN_GRP, apply_perturbation=True,
    use_bidir_norm=True, use_softmax_norm=False)
assert torch.isfinite(w_bd).all()
assert w_bd[:, :, :2].abs().max().item() < 1e-7
v_mean_bd = w_bd[valid].mean().item()
assert abs(v_mean_bd - 1.0) < 0.3
print(f"[6a] bidir=on  softmax=off : mean(w|valid)={v_mean_bd:.3f} ✓")

# (6b) softmax-only: c_t ≥ 0 strictly, and Σ_t c_t = T_valid per trajectory
w_sm, _ = compute_entropy_z_weights(
    entropy=entropy, valid_mask=valid, advantage=advantage,
    gid=gid_tsp, n_grp_per_inst=ngrp_tsp,
    gamma=0.3, min_group_size=MIN_GRP, apply_perturbation=True,
    use_bidir_norm=False, use_softmax_norm=True)
assert torch.isfinite(w_sm).all()
assert (w_sm >= 0).all(), "softmax mode must give non-negative weights"
T_valid_per_traj = valid.float().sum(dim=2)               # (B, P)
sum_per_traj     = w_sm.sum(dim=2)                         # (B, P)
err = (sum_per_traj - T_valid_per_traj).abs().max().item()
assert err < 1e-3, f"softmax conservation violated: max err={err}"
print(f"[6b] bidir=off softmax=on  : c_t≥0 ✓  Σc_t=T_valid (max err={err:.2e}) ✓")

# (6c) bidir + softmax: both invariants
w_both, _ = compute_entropy_z_weights(
    entropy=entropy, valid_mask=valid, advantage=advantage,
    gid=gid_tsp, n_grp_per_inst=ngrp_tsp,
    gamma=0.3, min_group_size=MIN_GRP, apply_perturbation=True,
    use_bidir_norm=True, use_softmax_norm=True)
assert (w_both >= 0).all()
sum_both = w_both.sum(dim=2)
err_both = (sum_both - T_valid_per_traj).abs().max().item()
assert err_both < 1e-3
print(f"[6c] bidir=on  softmax=on  : c_t≥0 ✓  Σc_t=T_valid (max err={err_both:.2e}) ✓")

# (6d) warmup with any combination → all valid c_t == 1
for bd in (False, True):
    for sm in (False, True):
        w_wm_x, _ = compute_entropy_z_weights(
            entropy=entropy, valid_mask=valid, advantage=advantage,
            gid=gid_tsp, n_grp_per_inst=ngrp_tsp,
            gamma=0.3, min_group_size=MIN_GRP, apply_perturbation=False,
            use_bidir_norm=bd, use_softmax_norm=sm)
        assert (w_wm_x[valid] - 1.0).abs().max().item() < 1e-4, \
            f"warmup w!=1 under bidir={bd} softmax={sm}"
        assert w_wm_x[~valid].abs().max().item() < 1e-7
print(f"[6d] warmup: all 4 (bidir, softmax) combos give valid w==1 ✓")


print("\n=== ALL SMOKE TESTS PASSED ===")
