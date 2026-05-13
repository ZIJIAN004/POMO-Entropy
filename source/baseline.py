"""
Pure group-wise entropy reweighting (no OLS, no MLP baseline).

Design philosophy
=================
Entropy = environment_effect(state) + model_confidence(state).
We want to extract pure model_confidence by subtracting the environment effect,
which we estimate as the group mean of entropy over "same-environment" steps.

"Same environment" is defined as same (instance, n_feasible[, at_depot,
load_bin, vis_ratio_bin]) — discrete features used directly, continuous
features (load, vis_ratio) split into equal-width bins.

Pipeline (per call):
    1. build_group_id(...)             — problem-aware dense gid construction
    2. group mean/std over entropy     — within-group, valid steps only
    3. ΔH = (entropy − grp_mean) / grp_std         (small groups → ΔH = 0)
    4. c_t = 1 + γ · sign(advantage) · ΔH_t        (warmup → c_t = 1)
    5. invalid steps → c_t = 0
    6. diagnostics: per-instance top3 concentration & small-group fraction
"""

import torch


# ---------------------------------------------------------------------------
# Group id construction — problem-aware. Returns (gid, n_grp_per_inst) where
# gid is (B, P, T) long in [0, n_grp_per_inst) and gid is per-instance (does
# NOT include batch offset; the reweighting routine adds that offset itself).
# ---------------------------------------------------------------------------

def build_group_id(problem_type, *, n_feasible, at_depot=None, load=None,
                   vis_ratio=None, n_bins=10):
    """Construct per-instance dense group id.

    TSP : gid = n_feasible                              (1 discrete dim)
    CVRP: gid = n_feasible · S₁ + at_depot · S₂
                + load_bin · n_bins + vis_ratio_bin     (1 disc + 1 bin + 1 disc + 1 bin)
          where S₂ = n_bins², S₁ = 2 · S₂.

    Continuous features are clamped into [0, 1] before binning (handles
    invalid/forced-step garbage values without crashing — they're masked
    out downstream anyway).
    """
    nf = n_feasible.long()                              # (B, P, T)

    if problem_type == 'tsp':
        return nf, int(nf.max().item()) + 1

    if problem_type == 'cvrp':
        assert at_depot is not None and load is not None and vis_ratio is not None, (
            "cvrp requires at_depot/load/vis_ratio for group construction")
        ad = at_depot.long()                            # (B, P, T) in {0,1}
        lb = (load.clamp(0.0, 1.0) * n_bins).long().clamp(max=n_bins - 1)
        vb = (vis_ratio.clamp(0.0, 1.0) * n_bins).long().clamp(max=n_bins - 1)
        max_nf = int(nf.max().item()) + 1
        S2 = n_bins * n_bins                            # stride for at_depot
        S1 = 2 * S2                                     # stride for n_feasible
        gid = nf * S1 + ad * S2 + lb * n_bins + vb
        return gid, max_nf * S1

    if problem_type == 'vrptw':
        # VRPTW: continuous features (current_time) are unnormalized; defer
        # proper handling. Reweighting on VRPTW is intentionally not supported
        # yet — caller must keep USE_ENTROPY_REWEIGHT off for vrptw.
        raise NotImplementedError(
            "entropy reweighting not yet supported for VRPTW — "
            "set USE_ENTROPY_REWEIGHT = False for this problem.")

    raise ValueError(f"Unknown problem_type: {problem_type}")


# ---------------------------------------------------------------------------
# Core reweighting: pure within-group z-score → linear perturbation c_t.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_entropy_z_weights(entropy, valid_mask, advantage, gid, n_grp_per_inst,
                               gamma, min_group_size=4, apply_perturbation=True):
    """
    entropy:           (B, P, T) — model entropy at each step
    valid_mask:        (B, P, T) bool — True = real decision step
    advantage:         (B, P)    — trajectory-level POMO advantage
    gid:               (B, P, T) long — per-instance group id in [0, n_grp_per_inst)
    n_grp_per_inst:    int       — number of distinct gid values per instance
    gamma:             float     — perturbation amplitude
    min_group_size:    int       — groups with fewer valid steps get ΔH = 0
    apply_perturbation: bool     — if False (warmup), forces ΔH = 0 everywhere
                                    but still computes monitoring stats.

    Returns:
        w:    (B, P, T) — c_t = 1 + γ·sign(A)·ΔH_t (or 1 during warmup),
                          invalid step = 0, small-group step = 1.
        diag: dict {
            'top3_concentration': scalar — mean over instances of
                                  sum(top-3 group sizes) / sum(all group sizes).
                                  Rises as policy/state homogenizes.
            'small_group_ratio': scalar — mean over instances of
                                  fraction of valid steps in groups with
                                  count < min_group_size.
        }
    """
    B, P, T = entropy.shape
    device = entropy.device

    # ── Flatten + add per-batch offset to gid so each instance gets its own
    #    block of [0, n_grp_per_inst) slots and groups never collide across
    #    instances. Total slots = B · n_grp_per_inst.
    gid_flat = gid.reshape(-1)                                       # (B·P·T,)
    bid = torch.arange(B, device=device).repeat_interleave(P * T)    # (B·P·T,)
    gid_global = bid * n_grp_per_inst + gid_flat                     # (B·P·T,)
    n_grp_total = B * n_grp_per_inst

    v_flat = valid_mask.reshape(-1).float()
    ent_flat = entropy.reshape(-1)

    # ── Group statistics over entropy (valid steps only). ─────────────────────
    counts = torch.zeros(n_grp_total, device=device).scatter_add_(0, gid_global, v_flat)
    sums   = torch.zeros(n_grp_total, device=device).scatter_add_(0, gid_global, ent_flat * v_flat)
    grp_mean = (sums / counts.clamp(min=1))[gid_global]

    centered = (ent_flat - grp_mean) * v_flat
    sq_sums  = torch.zeros(n_grp_total, device=device).scatter_add_(0, gid_global, centered.square())
    grp_std  = ((sq_sums / counts.clamp(min=1)).sqrt().clamp(min=1e-8))[gid_global]

    delta_H = (ent_flat - grp_mean) / grp_std                        # (B·P·T,)

    # ── Small-group mask: groups with too few valid samples get ΔH = 0. ───────
    sufficient_per_grp = (counts >= float(min_group_size)).float()   # (n_grp_total,)
    sufficient_per_step = sufficient_per_grp[gid_global]              # (B·P·T,)
    delta_H = delta_H * sufficient_per_step

    if not apply_perturbation:
        # Warmup: keep gid/counts for monitoring but force ΔH = 0 everywhere.
        delta_H = torch.zeros_like(delta_H)

    delta_H = delta_H.reshape(B, P, T)

    # ── Linear per-step perturbation c_t = 1 + γ·sign(A)·ΔH_t. ───────────────
    sign_A = advantage.sign().unsqueeze(2)                           # (B, P, 1)
    w = 1.0 + gamma * sign_A * delta_H                               # (B, P, T)
    w = w * valid_mask.float()                                       # invalid → 0

    # ── Diagnostics: per-instance top-3 concentration & small-group ratio. ────
    # counts is (B · n_grp_per_inst,) → reshape to (B, n_grp_per_inst).
    counts_per_inst = counts.reshape(B, n_grp_per_inst)
    total_per_inst = counts_per_inst.sum(dim=1).clamp(min=1.0)        # (B,)

    top3_per_inst = counts_per_inst.topk(min(3, n_grp_per_inst), dim=1).values.sum(dim=1)
    top3_concentration = (top3_per_inst / total_per_inst).mean()

    small_per_inst = (counts_per_inst *
                       (counts_per_inst < float(min_group_size)).float()).sum(dim=1)
    small_group_ratio = (small_per_inst / total_per_inst).mean()

    diag = {
        'top3_concentration': top3_concentration.detach(),
        'small_group_ratio':  small_group_ratio.detach(),
    }
    return w, diag
