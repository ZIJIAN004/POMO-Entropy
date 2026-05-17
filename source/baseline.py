"""
Pure group-wise entropy reweighting (no OLS, no MLP baseline).

Design philosophy
=================
Entropy = environment_effect(state) + model_confidence(state).
We want to extract pure model_confidence by subtracting the environment effect,
which we estimate as the group mean of entropy over "same-environment" steps.

"Same environment" is defined as same (instance, n_feasible[, at_depot,
load_bin/time_bin, vis_ratio_bin]) — discrete features used directly, continuous
features (load on CVRP, normalized current_time on VRPTW, vis_ratio) split into
equal-width bins.

Pipeline (per call):
    1. build_group_id(...)             — problem-aware dense gid construction
    2. (optional, use_bidir_norm)      — subtract per-trajectory mean from H
                                          BEFORE the bucket statistics. Removes
                                          α_i trajectory offset which drives
                                          within-bucket heteroscedasticity.
    3. group mean/std over (H or H_in) — within-group, valid steps only
    4. ΔH = (· − grp_mean) / grp_std    (small groups → ΔH = 0)
    5. c_t form (warmup → c_t = 1 on valid, 0 on invalid in both forms):
         use_softmax_norm=False (default):
             c_t = 1 + γ · sign(advantage) · ΔH_t                    [linear]
             — small-group (ΔH=0) and forced (n_feasible≤1, all H=0 in bucket
               → ΔH=0) steps automatically get c_t = 1.
         use_softmax_norm=True:
             rw_mask = valid & sufficient_bucket & (n_feasible > 1)
             — rw  steps: c_t = softmax(γ·sign(A)·ΔH over rw only) · T_rw
             — valid but not rw (small-group / forced): c_t = 1 (baseline)
             — invalid: c_t = 0
             — Isolates non-reweight steps from the softmax denominator so
               they neither influence nor are influenced by reweighted steps.
               Σ_t c_t = T_rw + (T_valid − T_rw) = T_valid preserved.
    6. invalid steps → c_t = 0 (linear: explicitly; softmax: torch.where)
    7. diagnostics: per-instance top3 concentration & small-group fraction
"""

import torch


# ---------------------------------------------------------------------------
# Group id construction — problem-aware. Returns (gid, n_grp_per_inst) where
# gid is (B, P, T) long in [0, n_grp_per_inst) and gid is per-instance (does
# NOT include batch offset; the reweighting routine adds that offset itself).
# ---------------------------------------------------------------------------

def build_group_id(problem_type, *, n_feasible, at_depot=None, load=None,
                   time=None, vis_ratio=None, n_bins=10):
    """Construct per-instance dense group id.

    TSP  : gid = n_feasible                              (1 discrete dim)
    CVRP : with vis_ratio (4-dim):
             gid = n_feasible · S₁ + at_depot · S₂
                   + load_bin · n_bins + vis_ratio_bin
             where S₂ = n_bins², S₁ = 2 · S₂.
           without vis_ratio (3-dim, vis_ratio=None):
             gid = n_feasible · (2 · n_bins) + at_depot · n_bins + load_bin
             → ~n_bins× fewer slots, larger buckets, more stable grp_std.
    VRPTW: same as CVRP but with normalized current_time replacing load.
           Caller must pre-normalize time into [0, 1] (e.g. current_time /
           tw_end_max). Resets at depot are fine — at_depot absorbs that.

    Continuous features are clamped into [0, 1] before binning (handles
    invalid/forced-step garbage values without crashing — they're masked
    out downstream anyway).
    """
    nf = n_feasible.long()                              # (B, P, T)

    if problem_type == 'tsp':
        return nf, int(nf.max().item()) + 1

    if problem_type in ('cvrp', 'vrptw'):
        if problem_type == 'cvrp':
            assert at_depot is not None and load is not None, (
                "cvrp requires at_depot/load for group construction")
            cont = load
        else:
            assert at_depot is not None and time is not None, (
                "vrptw requires at_depot/time for group construction")
            cont = time
        ad = at_depot.long()                            # (B, P, T) in {0,1}
        cb = (cont.clamp(0.0, 1.0) * n_bins).long().clamp(max=n_bins - 1)
        max_nf = int(nf.max().item()) + 1
        if vis_ratio is not None:
            vb = (vis_ratio.clamp(0.0, 1.0) * n_bins).long().clamp(max=n_bins - 1)
            S2 = n_bins * n_bins                        # stride for at_depot
            S1 = 2 * S2                                 # stride for n_feasible
            gid = nf * S1 + ad * S2 + cb * n_bins + vb
            return gid, max_nf * S1
        # 3-dim variant: drop vis_ratio dimension entirely.
        S1 = 2 * n_bins                                 # stride for n_feasible
        gid = nf * S1 + ad * n_bins + cb
        return gid, max_nf * S1

    raise ValueError(f"Unknown problem_type: {problem_type}")


# ---------------------------------------------------------------------------
# Core reweighting: pure within-group z-score → linear perturbation c_t.
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_entropy_z_weights(entropy, valid_mask, advantage, gid, n_grp_per_inst,
                               gamma, min_group_size=4, apply_perturbation=True,
                               use_bidir_norm=False, use_softmax_norm=False,
                               n_feasible=None, low_grp_mean_thresh=0.0):
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
    use_bidir_norm:    bool      — if True, subtract per-trajectory mean from
                                    entropy BEFORE the (instance, gid) z-score.
                                    Removes the α_i trajectory offset that
                                    drives within-bucket heteroscedasticity.
    use_softmax_norm:  bool      — if True, isolate small-group and forced
                                    (n_feasible≤1) steps from softmax: those
                                    keep c_t=1, only reweight-eligible steps
                                    enter softmax(γ·sign(A)·ΔH) · T_rw.
                                    Preserves c_t≥0, Σ_t c_t = T_valid, and
                                    matches the linear form's baseline behavior
                                    on non-reweight steps. If False, linear
                                    c_t = 1+γ·…
    n_feasible:        (B, P, T) — feasible action count per step. Required
                                    when use_softmax_norm=True (used to mark
                                    deterministic steps as non-reweight);
                                    ignored otherwise.
    low_grp_mean_thresh: float    — buckets whose grp_mean < this are stripped
                                     from rw_mask (set to baseline c_t=1 under
                                     softmax, c_t=1+0=1 under linear since their
                                     ΔH is left untouched in the perturbation
                                     branch but they're held outside rw). 0.0
                                     disables the filter. Use ~0.05 to drop
                                     near-deterministic buckets where ΔH is
                                     essentially noise.

    Returns:
        w:    (B, P, T) — c_t. invalid step = 0; small-group step = 1
                          (linear mode) or uniform within softmax (softmax mode).
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
    vmf = valid_mask.float()

    # ── (Optional) Bidirectional normalization: subtract per-trajectory mean.
    #    Removes α_i so the subsequent (instance, gid) z-score sees only
    #    within-bucket variance free of the trajectory offset.
    if use_bidir_norm:
        cnt_t = vmf.sum(dim=2, keepdim=True).clamp(min=1.0)           # (B, P, 1)
        mu_t  = (entropy * vmf).sum(dim=2, keepdim=True) / cnt_t      # (B, P, 1)
        H_in  = entropy - mu_t                                         # (B, P, T)
    else:
        H_in  = entropy

    # ── Flatten + add per-batch offset to gid so each instance gets its own
    #    block of [0, n_grp_per_inst) slots and groups never collide across
    #    instances. Total slots = B · n_grp_per_inst.
    gid_flat = gid.reshape(-1)                                       # (B·P·T,)
    bid = torch.arange(B, device=device).repeat_interleave(P * T)    # (B·P·T,)
    gid_global = bid * n_grp_per_inst + gid_flat                     # (B·P·T,)
    n_grp_total = B * n_grp_per_inst

    v_flat = vmf.reshape(-1)
    ent_flat = H_in.reshape(-1)

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
    suff_step = sufficient_per_step.view(B, P, T).bool()             # (B, P, T)

    # Reweight-eligible: valid AND in a sufficient bucket AND not forced.
    # Computed once and reused by softmax c_t and the rw_ratio diagnostic.
    # If n_feasible is not provided (linear-only callers), we fall back to
    # (valid & sufficient) — same shape, used purely for monitoring.
    if n_feasible is not None:
        rw_mask = valid_mask & suff_step & (n_feasible > 1)          # (B, P, T) bool
    else:
        rw_mask = valid_mask & suff_step

    # Low-entropy bucket filter: buckets whose grp_mean is below the threshold
    # carry essentially noise in ΔH (near-deterministic policy). Strip them
    # from rw_mask → softmax: c_t=1; linear: ΔH still = z(noise) but since the
    # caller can drive ΔH→0 by combining with sufficient_per_step downstream,
    # we just zero ΔH on the dropped steps too for safety.
    if low_grp_mean_thresh > 0.0:
        grp_mean_step = grp_mean.view(B, P, T)
        keep_grp = grp_mean_step > float(low_grp_mean_thresh)         # (B, P, T)
        rw_mask = rw_mask & keep_grp
        # Zero ΔH on the filtered-out steps so the linear branch also leaves
        # them at c_t = 1 + γ·sign(A)·0 = 1.
        delta_H = delta_H * keep_grp.float()

    # ── c_t: linear (1+γ·sign(A)·ΔH) or isolated softmax over rw subset ──────
    sign_A = advantage.sign().unsqueeze(2)                           # (B, P, 1)
    if use_softmax_norm:
        assert n_feasible is not None, (
            "use_softmax_norm=True requires n_feasible (B,P,T) to identify "
            "forced (n_feasible≤1) steps that must stay at c_t=1.")

        # Small-group (suff_step=False) and forced (n_feasible≤1) steps are
        # held at c_t=1 — isolated from the softmax denominator so they
        # neither dilute the redistribution nor get diluted by it.
        logit = (gamma * sign_A * delta_H).masked_fill(~rw_mask, -1e9)
        T_rw  = rw_mask.float().sum(dim=2, keepdim=True).clamp(min=1.0)
        w_rw  = torch.softmax(logit, dim=2) * T_rw                    # softmax over rw only

        # Compose: invalid → 0; valid & not rw → 1; rw → softmax weight.
        w = torch.where(rw_mask, w_rw, vmf)
    else:
        w = 1.0 + gamma * sign_A * delta_H                            # (B, P, T)
        w = w * vmf                                                    # invalid → 0

    # ── Diagnostics: per-instance top-3 concentration & small-group ratio. ────
    # counts is (B · n_grp_per_inst,) → reshape to (B, n_grp_per_inst).
    counts_per_inst = counts.reshape(B, n_grp_per_inst)
    total_per_inst = counts_per_inst.sum(dim=1).clamp(min=1.0)        # (B,)

    top3_per_inst = counts_per_inst.topk(min(3, n_grp_per_inst), dim=1).values.sum(dim=1)
    top3_concentration = (top3_per_inst / total_per_inst).mean()

    small_per_inst = (counts_per_inst *
                       (counts_per_inst < float(min_group_size)).float()).sum(dim=1)
    small_group_ratio = (small_per_inst / total_per_inst).mean()

    # rw_ratio: per-instance fraction of valid steps that pass into softmax
    # reweighting (≡ valid & sufficient_bucket & n_feasible>1 when n_feasible
    # given; ≡ valid & sufficient_bucket otherwise). Low rw_ratio = most steps
    # are baseline-treated, reweighting has little leverage on the trajectory.
    rw_per_inst    = rw_mask.float().sum(dim=(1, 2))                  # (B,)
    valid_per_inst = vmf.sum(dim=(1, 2)).clamp(min=1.0)               # (B,)
    rw_ratio = (rw_per_inst / valid_per_inst).mean()

    diag = {
        'top3_concentration': top3_concentration.detach(),
        'small_group_ratio':  small_group_ratio.detach(),
        'rw_ratio':           rw_ratio.detach(),
        'delta_H':            delta_H.detach(),                          # (B, P, T)
        'rw_mask':            rw_mask.detach(),                          # (B, P, T) bool
        'grp_mean':           grp_mean.view(B, P, T).detach(),           # bucket mean of H_in,
                                                                          # per-step broadcast.
                                                                          # When use_bidir_norm=False
                                                                          # this equals raw H bucket mean.
    }
    return w, diag
