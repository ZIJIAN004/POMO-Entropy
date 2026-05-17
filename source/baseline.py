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

    # R²_grp: per-instance share of H_in variance explained by the bucket grouping.
    # R² = 1 − within_ss / total_ss, where:
    #   within_ss = Σ_g Σ_{i∈g} (H_in_i − grp_mean_g)²    (= sq_sums.sum())
    #   total_ss  = Σ_i (H_in_i − global_mean_i)²
    # Tells whether the partition is actually homogenizing entropy:
    #   ≈ 0  : buckets are no better than random — partition wrong
    #   .3–.5: bucket explains main variation, residual noise (healthy)
    #   .6–.8: very clean buckets
    #   ≈ 1  : likely over-fit (each bucket has 1 step)
    H_sq_per_inst  = (H_in * H_in * vmf).sum(dim=(1, 2))              # (B,)
    H_sum_per_inst = (H_in * vmf).sum(dim=(1, 2))                     # (B,)
    n_per_inst     = vmf.sum(dim=(1, 2)).clamp(min=1.0)               # (B,)
    total_ss_per_inst  = (H_sq_per_inst
                          - H_sum_per_inst.square() / n_per_inst)     # (B,)
    within_ss_per_inst = sq_sums.reshape(B, n_grp_per_inst).sum(dim=1) # (B,)
    r2_per_inst = 1.0 - within_ss_per_inst / total_ss_per_inst.clamp(min=1e-8)
    r2_grp = r2_per_inst.clamp(min=0.0, max=1.0).mean()

    diag = {
        'top3_concentration': top3_concentration.detach(),
        'small_group_ratio':  small_group_ratio.detach(),
        'rw_ratio':           rw_ratio.detach(),
        'r2_grp':             r2_grp.detach(),                            # bucket-variance-explained
        'delta_H':            delta_H.detach(),                          # (B, P, T)
        'rw_mask':            rw_mask.detach(),                          # (B, P, T) bool
        'grp_mean':           grp_mean.view(B, P, T).detach(),           # bucket mean of H_in,
                                                                          # per-step broadcast.
                                                                          # When use_bidir_norm=False
                                                                          # this equals raw H bucket mean.
    }
    return w, diag


# ---------------------------------------------------------------------------
# Monotonic-segment baseline: trajectory-internal contrast, no bucketing.
#
# Idea: instead of estimating environment_effect from "same state" cohorts
# (which fails on CVRP/VRPTW because state→entropy is non-smooth), use the
# trajectory's own local trend as anchor.
#
#   • Identify reversal points: t where sign(H_t − H_{t-1}) flips vs the
#     previous diff. Each reversal closes the prior monotone run.
#   • For each step t, anchor[t] = index of the most recent reversal at or
#     before t (or step 0 if no reversal yet).
#   • ΔH_local[t] = H_t − H[anchor[t]]
#     — positive: trajectory is in an "entropy rising" segment at step t
#     — negative: in an "entropy falling" segment
#     — magnitude = how far the current monotone run has carried H
#   • c_t = softmax over trajectory of (γ · sign(A) · ΔH_local), scaled to
#     preserve Σ_t c_t = T_rw on rw steps. Valid-but-not-rw → c_t=1.
#
# No environment estimation needed. Tradeoffs:
#   + zero bin/state similarity assumptions
#   + every trajectory has its own anchor, no cross-instance noise
#   − only captures "local trend" signal; misses absolute level info
#   − early steps have weak signal (anchor is just step 0)
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_monoseg_weights(entropy, valid_mask, advantage,
                             gamma, n_feasible=None,
                             apply_perturbation=True):
    """
    entropy:           (B, P, T) — model entropy per step
    valid_mask:        (B, P, T) bool — True = real decision step
    advantage:         (B, P)    — trajectory-level POMO advantage
    gamma:             float     — perturbation amplitude in softmax logit
    n_feasible:        (B, P, T) — feasible action count; forced steps
                                    (n_feasible ≤ 1) excluded from rw_mask
    apply_perturbation: bool     — warmup → c_t = 1 on valid, monitor only

    Returns:
        w:    (B, P, T) — invalid=0, valid-non-rw=1, rw=softmax-weighted (Σ=T_rw)
        diag: dict {
            'delta_H_local'      : (B, P, T) — segment-anchored Δ
            'rw_mask'            : (B, P, T) bool
            'rw_ratio'           : scalar
            'n_segs_per_traj'    : scalar — mean number of monotone segments per traj
            'seg_len_mean'       : scalar — mean monotone-run length (in steps)
            'delta_pos_mean'     : scalar — mean ΔH_local over rising-segment steps
            'delta_neg_mean'     : scalar — mean ΔH_local over falling-segment steps
            'ct_top3'            : scalar — softmax c_t top-3 concentration per traj
        }
    """
    B, P, T = entropy.shape
    device = entropy.device
    vmf = valid_mask.float()

    # ── 1. First difference and its sign per (B, P, T). ──────────────────────
    # diff[t] = H[t] - H[t-1] for t >= 1; diff[0] = 0.
    diff = torch.zeros_like(entropy)
    diff[:, :, 1:] = entropy[:, :, 1:] - entropy[:, :, :-1]
    sd = diff.sign()                                              # ±1 or 0

    # ── 2. Reversal indicator: sign changes between consecutive diffs. ───────
    # reversal[t] = True iff sd[t] and sd[t-1] are both non-zero and opposite.
    # Treat 0 (flat segment) as "continues previous direction" → no reversal.
    # reversal_at_step[t] in (B,P,T): a reversal occurred at step t (t >= 2).
    reversal_at_step = torch.zeros(B, P, T, dtype=torch.bool, device=device)
    if T >= 3:
        sd_t   = sd[:, :, 2:]              # (B, P, T-2)  — sign(H_t - H_{t-1})
        sd_tm1 = sd[:, :, 1:-1]            # (B, P, T-2)  — sign(H_{t-1} - H_{t-2})
        flip = (sd_t * sd_tm1 < 0)         # both non-zero AND opposite signs
        reversal_at_step[:, :, 2:] = flip
    # The "anchor" we want is the position where the NEW trend STARTS, which
    # is the step BEFORE the reversal-detected step (the local extremum):
    #   if reversal_at_step[t] = True, then anchor for t..next_rev-1 is t-1.
    # Implement via cumulative max over (step_idx if reversal else 0), then
    # subtract 1 to get the extremum's index — but for t=0,1 we must keep 0.
    step_idx = torch.arange(T, device=device).view(1, 1, T).expand(B, P, T)
    rev_step = torch.where(reversal_at_step,
                            (step_idx - 1).clamp(min=0),
                            torch.zeros_like(step_idx))
    anchor = rev_step.cummax(dim=2).values                       # (B, P, T) long

    # ── 3. ΔH_local = H_t − H[anchor[t]]. ────────────────────────────────────
    H_at_anchor = torch.gather(entropy, dim=2, index=anchor)
    delta_H_local = entropy - H_at_anchor                         # (B, P, T)

    if not apply_perturbation:
        delta_H_local = torch.zeros_like(delta_H_local)

    # ── 4. rw_mask: valid & not forced. No bucket-sufficiency needed. ────────
    if n_feasible is not None:
        rw_mask = valid_mask & (n_feasible > 1)
    else:
        rw_mask = valid_mask.clone()

    # ── 5. softmax c_t over trajectory (rw subset). ─────────────────────────
    sign_A = advantage.sign().unsqueeze(2)                        # (B, P, 1)
    logit = (gamma * sign_A * delta_H_local).masked_fill(~rw_mask, -1e9)
    T_rw = rw_mask.float().sum(dim=2, keepdim=True).clamp(min=1.0)
    w_rw = torch.softmax(logit, dim=2) * T_rw

    # Compose: invalid → 0; valid & not rw → 1; rw → softmax weight.
    w = torch.where(rw_mask, w_rw, vmf)

    # ── 6. Diagnostics. ─────────────────────────────────────────────────────
    # rw coverage
    rw_per_inst    = rw_mask.float().sum(dim=(1, 2))
    valid_per_inst = vmf.sum(dim=(1, 2)).clamp(min=1.0)
    rw_ratio = (rw_per_inst / valid_per_inst).mean()

    # segments per trajectory: count reversal_at_step among valid steps + 1
    # (first segment has no leading reversal, so add 1 if any valid steps exist)
    rev_valid = reversal_at_step & valid_mask
    n_segs_per_traj = rev_valid.float().sum(dim=2).mean() + 1.0

    # mean segment length = T_valid / n_segs
    T_valid_per_traj = valid_mask.float().sum(dim=2).clamp(min=1.0)        # (B, P)
    n_segs_per_xtraj = rev_valid.float().sum(dim=2) + 1.0                   # (B, P)
    seg_len_mean = (T_valid_per_traj / n_segs_per_xtraj).mean()

    # ΔH_local mean on rising/falling rw steps
    pos_mask = rw_mask & (delta_H_local > 0)
    neg_mask = rw_mask & (delta_H_local < 0)
    delta_pos_mean = (delta_H_local * pos_mask.float()).sum() / pos_mask.float().sum().clamp(min=1.0)
    delta_neg_mean = (delta_H_local * neg_mask.float()).sum() / neg_mask.float().sum().clamp(min=1.0)

    # c_t top-3 concentration per trajectory: how peaky is the softmax?
    # Higher = the trajectory's weight is concentrated on a few decisive steps.
    ct_top3_per_traj = w_rw.topk(min(3, T), dim=2).values.sum(dim=2)        # (B, P)
    ct_top3 = (ct_top3_per_traj / T_rw.squeeze(2).clamp(min=1.0)).mean()

    diag = {
        'delta_H_local':   delta_H_local.detach(),
        'rw_mask':         rw_mask.detach(),
        'rw_ratio':        rw_ratio.detach(),
        'n_segs_per_traj': n_segs_per_traj.detach(),
        'seg_len_mean':    seg_len_mean.detach(),
        'delta_pos_mean':  delta_pos_mean.detach(),
        'delta_neg_mean':  delta_neg_mean.detach(),
        'ct_top3':         ct_top3.detach(),
    }
    return w, diag
