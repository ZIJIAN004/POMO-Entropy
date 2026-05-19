"""
Train.py — POMO REINFORCE with optional pure group-wise entropy reweighting.

Supported modes:
  • USE_ENTROPY_REWEIGHT = False : baseline POMO (no reweighting)
  • USE_ENTROPY_REWEIGHT = True  : per-step c_t reweighting where ΔH is the
                                   within-group z-score of entropy over
                                   (instance, n_feasible[, at_depot, load_bin,
                                   vis_ratio_bin]).
                                   First ENTROPY_WARMUP_EPOCHS epochs run with
                                   ΔH = 0 (baseline POMO) but still log
                                   monitoring stats.
        Ablation switches (orthogonal, 4 combinations):
          USE_BIDIR_NORM   : subtract per-trajectory mean from entropy BEFORE
                             the bucket z-score; addresses within-bucket
                             heteroscedasticity caused by α_i offset.
          USE_SOFTMAX_NORM : rw_mask = valid & sufficient_bucket &
                             (n_feasible > 1). rw steps → c_t = softmax(γ·
                             sign(A)·ΔH over rw only) · T_rw; small-group /
                             forced (non-rw valid) → c_t = 1 (isolated from
                             softmax denominator); invalid → 0. Preserves
                             Σ_t c_t = T_valid and matches the linear form
                             on baseline-treated steps. Without it, the
                             linear c_t = 1 + γ·sign(A)·ΔH is used.

  • USE_ENTROPY_BONUS  : A2C-style entropy bonus (independent, can stack with
                         the reweighting path).

Excluded from the group statistics & from the gradient via valid_mask:
  - forced steps  (TSP: step 0; CVRP/VRPTW: steps 0-1)
  - finished padding steps (CVRP/VRPTW)

advantage_i = r_i - mean_j(r_j)
"""

import time
import torch

from HYPER_PARAMS import *
from source.utilities import Average_Meter
from source.baseline import (build_group_id, compute_entropy_z_weights,
                              compute_monoseg_weights,
                              compute_trajinternal_weights,
                              anova_omega_squared, MLPSignalEstimator)


# Persistent MLP estimator across batches and epochs (one per process).
# Created lazily on first batch since we need feature dim from runtime.
_MLP_ESTIMATOR = None
_MLP_DEVICE = None


def _make_valid_mask(T_total, batch_size, device, finished_list=None):
    """forced-steps + finished padding → invalid; rest valid."""
    valid = torch.ones(batch_size, POMO_SIZE, T_total, dtype=torch.bool, device=device)
    forced_steps = 1 if PROBLEM_TYPE == 'tsp' else 2
    if T_total > forced_steps:
        valid[:, :, :forced_steps] = False
    else:
        valid[:] = False
    if finished_list is not None:
        valid &= ~finished_list
    return valid


def TRAIN(model, env, optimizer, lr_scheduler, epoch, timer_start, logger):
    model.train()

    score_AM = Average_Meter()
    loss_AM  = Average_Meter()
    # ── Kept (pipeline-health) ──────────────────────────────────────────────
    top3_AM    = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # bucket size concentration
    rw_AM      = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # rw_mask coverage
    ctTop3_AM  = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # c_t softmax concentration
    # ── New H1-H4 diagnostics ───────────────────────────────────────────────
    omega_AM   = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # H2 — bucket ANOVA ω²
    mlpR2_AM   = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # H1 — MLP held-out R² (signal upper bound)
    lagCorr_AM = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # H1 — corr(H_t, H_{t-1}) lag-1 autocorrelation
    ctCV_AM    = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # H3 — c_t coefficient of variation
    sigRat_AM  = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # H4 — σ²_traj / σ²_step (signal granularity)
    # ── Monoseg-only extras kept for backward compat ────────────────────────
    nsegs_AM   = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # monoseg
    seglen_AM  = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # monoseg

    logger_start = time.time()
    episode      = 0
    device       = next(model.parameters()).device

    # True once warmup is done — apply the actual perturbation. During warmup
    # we still compute group stats (for monitoring) but force ΔH = 0.
    apply_pert = USE_ENTROPY_REWEIGHT and (epoch > ENTROPY_WARMUP_EPOCHS)

    # Collect entropy + group-construction features only when reweighting is on.
    collect_groups   = USE_ENTROPY_REWEIGHT or USE_ENTROPY_BONUS
    needs_cvrp_feats  = (USE_ENTROPY_REWEIGHT and PROBLEM_TYPE == 'cvrp')
    needs_vrptw_feats = (USE_ENTROPY_REWEIGHT and PROBLEM_TYPE == 'vrptw')
    needs_route_feats = needs_cvrp_feats or needs_vrptw_feats

    while episode < TRAIN_EPISODES:
        batch_size = min(TRAIN_BATCH_SIZE, TRAIN_EPISODES - episode)
        episode   += batch_size

        # ── Rollout ──────────────────────────────────────────────────────
        env.load_problems(batch_size, device=device)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        prob_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if collect_groups:
            entropy_list    = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
            n_feasible_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if needs_route_feats:
            at_depot_list  = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
            vis_ratio_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if needs_cvrp_feats:
            load_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if needs_vrptw_feats:
            time_list  = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
            # per-instance normalizer for current_time: max customer tw_end.
            # depot tw_end = 1e9 by construction; excluded via [:, 1:].
            time_norm = env.tw_end[:, 1:].max(dim=1).values.clamp(min=1e-6)  # (B,)
            time_norm = time_norm[:, None]                                    # (B, 1)
        if needs_route_feats:
            finished_list = torch.zeros(batch_size, POMO_SIZE, 0,
                                         dtype=torch.bool, device=device)
        else:
            finished_list = None

        state, reward, done = env.pre_step()

        while not done:
            selected, prob, entropy, margin = model(state)

            # Pick which scalar to use as "confidence signal" for reweight.
            if CONFIDENCE_SIGNAL == 'margin':
                signal = margin
            else:
                signal = entropy

            if collect_groups:
                # entropy_list keeps its name for backward compatibility, but
                # may now hold prob margin depending on CONFIDENCE_SIGNAL.
                entropy_list = torch.cat(
                    (entropy_list, signal[:, :, None]), dim=2)
                n_feasible_list = torch.cat(
                    (n_feasible_list, state.n_feasible[:, :, None].float()), dim=2)

            if needs_route_feats:
                if state.current_node is None:
                    at_depot = torch.zeros(batch_size, POMO_SIZE, device=device)
                else:
                    at_depot = (state.current_node == 0).float()
                at_depot_list = torch.cat(
                    (at_depot_list, at_depot[:, :, None]), dim=2)

                vis_step = state.visited_customer_count.float() / max(PROBLEM_SIZE, 1)
                vis_ratio_list = torch.cat(
                    (vis_ratio_list, vis_step[:, :, None]), dim=2)

            if needs_cvrp_feats:
                load_list = torch.cat(
                    (load_list, state.load[:, :, None]), dim=2)

            if needs_vrptw_feats:
                # current_time is reset to 0 at depot; max customer tw_end caps it.
                t_step = state.current_time / time_norm                  # (B, P) in ~[0,1]
                time_list = torch.cat(
                    (time_list, t_step[:, :, None]), dim=2)

            if finished_list is not None:
                finished_list = torch.cat(
                    (finished_list, state.finished[:, :, None]), dim=2)

            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # ── REINFORCE advantage ──────────────────────────────────────────
        reward_f  = reward.float()
        advantage = reward_f - reward_f.mean(dim=1, keepdim=True)

        # ── Compute weighted log_prob ────────────────────────────────────
        if USE_ENTROPY_REWEIGHT:
            T_total = prob_list.size(2)
            valid   = _make_valid_mask(T_total, batch_size, device, finished_list)

            if USE_TRAJINTERNAL_BASELINE:
                # Pure trajectory-internal: ΔH = H − μ_traj, softmax over all
                # valid steps, Σ_t c_t = T_valid. No bucket, no env estimation.
                weights, diag = compute_trajinternal_weights(
                    entropy=entropy_list,
                    valid_mask=valid,
                    advantage=advantage,
                    gamma=ENTROPY_GAMMA,
                    n_feasible=n_feasible_list,
                    apply_perturbation=apply_pert,
                )
                log_prob = (prob_list.log() * weights).sum(dim=2)

                if rw_AM is not None:
                    rw_AM.push(diag['rw_ratio'].unsqueeze(0))
                if ctTop3_AM is not None:
                    ctTop3_AM.push(diag['ct_top3'].unsqueeze(0))
            elif USE_MONOSEG_BASELINE:
                # Trajectory-internal monotonic-segment baseline.
                # Optional post-bucket norm: 3-dim bucket (no vis_ratio).
                gid_arg, n_grp_arg = None, None
                if USE_MONOSEG_POSTBUCKET:
                    if PROBLEM_TYPE == 'tsp':
                        gid_arg, n_grp_arg = build_group_id(
                            'tsp', n_feasible=n_feasible_list,
                            n_bins=ENTROPY_N_BINS)
                    elif PROBLEM_TYPE == 'cvrp':
                        gid_arg, n_grp_arg = build_group_id(
                            'cvrp',
                            n_feasible=n_feasible_list,
                            at_depot=at_depot_list,
                            load=load_list,
                            vis_ratio=None,          # postbucket forces 3-dim
                            n_bins=ENTROPY_N_BINS,
                        )
                    else:  # vrptw
                        gid_arg, n_grp_arg = build_group_id(
                            'vrptw',
                            n_feasible=n_feasible_list,
                            at_depot=at_depot_list,
                            time=time_list,
                            vis_ratio=None,
                            n_bins=ENTROPY_N_BINS,
                        )

                weights, diag = compute_monoseg_weights(
                    entropy=entropy_list,
                    valid_mask=valid,
                    advantage=advantage,
                    gamma=ENTROPY_GAMMA,
                    n_feasible=n_feasible_list,
                    apply_perturbation=apply_pert,
                    gid=gid_arg,
                    n_grp_per_inst=n_grp_arg,
                    min_group_size=ENTROPY_MIN_GROUP_SIZE,
                )
                log_prob = (prob_list.log() * weights).sum(dim=2)

                if rw_AM is not None:
                    rw_AM.push(diag['rw_ratio'].unsqueeze(0))
                if nsegs_AM is not None:
                    nsegs_AM.push(diag['n_segs_per_traj'].unsqueeze(0))
                if seglen_AM is not None:
                    seglen_AM.push(diag['seg_len_mean'].unsqueeze(0))
                if ctTop3_AM is not None:
                    ctTop3_AM.push(diag['ct_top3'].unsqueeze(0))
            else:
                # USE_VIS_RATIO_BIN=False → 3-dim bucket (drop vis_ratio).
                # vis_ratio_list only exists on CVRP/VRPTW (TSP doesn't collect it).
                if PROBLEM_TYPE == 'tsp':
                    gid, n_grp = build_group_id(
                        'tsp', n_feasible=n_feasible_list, n_bins=ENTROPY_N_BINS)
                elif PROBLEM_TYPE == 'cvrp':
                    vis_arg = vis_ratio_list if USE_VIS_RATIO_BIN else None
                    gid, n_grp = build_group_id(
                        'cvrp',
                        n_feasible=n_feasible_list,
                        at_depot=at_depot_list,
                        load=load_list,
                        vis_ratio=vis_arg,
                        n_bins=ENTROPY_N_BINS,
                    )
                else:  # vrptw
                    vis_arg = vis_ratio_list if USE_VIS_RATIO_BIN else None
                    gid, n_grp = build_group_id(
                        'vrptw',
                        n_feasible=n_feasible_list,
                        at_depot=at_depot_list,
                        time=time_list,
                        vis_ratio=vis_arg,
                        n_bins=ENTROPY_N_BINS,
                    )

                weights, diag = compute_entropy_z_weights(
                    entropy=entropy_list,
                    valid_mask=valid,
                    advantage=advantage,
                    gid=gid,
                    n_grp_per_inst=n_grp,
                    gamma=ENTROPY_GAMMA,
                    min_group_size=ENTROPY_MIN_GROUP_SIZE,
                    apply_perturbation=apply_pert,
                    use_bidir_norm=USE_BIDIR_NORM,
                    use_softmax_norm=USE_SOFTMAX_NORM,
                    n_feasible=n_feasible_list,
                    low_grp_mean_thresh=LOW_GRP_MEAN_THRESH,
                    use_robust_norm=USE_ROBUST_NORM,
                )
                log_prob = (prob_list.log() * weights).sum(dim=2)

                if top3_AM is not None:
                    top3_AM.push(diag['top3_concentration'].unsqueeze(0))
                if rw_AM is not None:
                    rw_AM.push(diag['rw_ratio'].unsqueeze(0))

                # c_t concentration (H3 in bucket path — top-3 c_t share per traj)
                with torch.no_grad():
                    rwm_f = diag['rw_mask'].float()
                    w_rw_only = weights * rwm_f
                    T_rw_per = rwm_f.sum(dim=2).clamp(min=1.0)               # (B, P)
                    ct_top3_per = w_rw_only.topk(min(3, w_rw_only.size(2)), dim=2).values.sum(dim=2)
                    ct_top3 = (ct_top3_per / T_rw_per).mean()
                    if ctTop3_AM is not None:
                        ctTop3_AM.push(ct_top3.unsqueeze(0))

            # ── H1-H4 diagnostic suite (shared across monoseg/bucket paths) ─
            # Build 4-d (CVRP/VRPTW) or 1-d (TSP) state features (normalized).
            if PROBLEM_TYPE == 'tsp':
                feats_main = (n_feasible_list / float(POMO_SIZE)).unsqueeze(-1)
            elif PROBLEM_TYPE == 'cvrp':
                feats_main = torch.stack([
                    n_feasible_list / float(POMO_SIZE),
                    at_depot_list, load_list, vis_ratio_list,
                ], dim=-1)
            else:   # vrptw
                feats_main = torch.stack([
                    n_feasible_list / float(POMO_SIZE),
                    at_depot_list, time_list, vis_ratio_list,
                ], dim=-1)

            in_dim      = feats_main.size(-1)
            feats_flat  = feats_main.reshape(-1, in_dim)
            signal_flat = entropy_list.reshape(-1)        # actually carries chosen CONFIDENCE_SIGNAL
            valid_flat  = valid.float().reshape(-1)

            # ── H1: MLP held-out R² — signal's state-predictable upper bound.
            # If even the MLP can't fit signal | state above ~0.1, the signal
            # itself has little state-predictable structure → step-level
            # reweight has no leverage regardless of method.
            global _MLP_ESTIMATOR, _MLP_DEVICE
            if _MLP_ESTIMATOR is None or _MLP_DEVICE != device:
                _MLP_ESTIMATOR = MLPSignalEstimator(in_dim=in_dim).to(device)
                _MLP_DEVICE = device
            r2_mlp = _MLP_ESTIMATOR.step_and_eval(
                feats_flat, signal_flat, valid_flat)
            if mlpR2_AM is not None:
                mlpR2_AM.push(r2_mlp.unsqueeze(0))

            with torch.no_grad():
                # ── H2: ANOVA ω² — objective bucket quality, no feature form
                # dependence. Cohen: < 0.01 negligible, 0.06 medium, 0.14 large.
                if PROBLEM_TYPE == 'tsp':
                    gid_d, n_grp_d = build_group_id(
                        'tsp', n_feasible=n_feasible_list, n_bins=ENTROPY_N_BINS)
                elif PROBLEM_TYPE == 'cvrp':
                    gid_d, n_grp_d = build_group_id(
                        'cvrp', n_feasible=n_feasible_list,
                        at_depot=at_depot_list, load=load_list,
                        vis_ratio=vis_ratio_list, n_bins=ENTROPY_N_BINS)
                else:
                    gid_d, n_grp_d = build_group_id(
                        'vrptw', n_feasible=n_feasible_list,
                        at_depot=at_depot_list, time=time_list,
                        vis_ratio=vis_ratio_list, n_bins=ENTROPY_N_BINS)
                bid_d = torch.arange(batch_size, device=device).repeat_interleave(
                    POMO_SIZE * gid_d.size(-1))
                gid_global_d = bid_d * n_grp_d + gid_d.reshape(-1)
                omega_sq, _ = anova_omega_squared(
                    signal_flat, valid_flat, gid_global_d, batch_size * n_grp_d)
                if omega_AM is not None:
                    omega_AM.push(omega_sq.unsqueeze(0))

                # ── H3: c_t coefficient of variation. Low CV → softmax is
                # nearly uniform → reweight effectively does nothing.
                vmf = valid.float()
                n_valid = vmf.sum().clamp(min=1.0)
                ct_mean = (weights * vmf).sum() / n_valid
                ct_var  = (((weights - ct_mean) * vmf) ** 2).sum() / n_valid
                ct_cv   = ct_var.sqrt() / ct_mean.clamp(min=1e-8)
                if ctCV_AM is not None:
                    ctCV_AM.push(ct_cv.unsqueeze(0))

                # ── H4: σ²_traj / σ²_step — signal's natural granularity.
                # > 1 → signal varies more across trajectories than within
                # (signal is trajectory-level, step reweight is wrong unit).
                # < 1 → step-level variation dominates (step reweight is right
                # unit for granularity, regardless of whether it works).
                n_per_traj  = vmf.sum(dim=2).clamp(min=1.0)                      # (B, P)
                mean_per_t  = (entropy_list * vmf).sum(dim=2) / n_per_traj      # (B, P)
                sig_traj    = mean_per_t.var(dim=1, unbiased=False).mean()
                diff_t      = entropy_list - mean_per_t.unsqueeze(2)
                sig_step_per = ((diff_t ** 2) * vmf).sum(dim=2) / n_per_traj
                sig_step    = sig_step_per.mean()
                sig_ratio   = sig_traj / sig_step.clamp(min=1e-8)
                if sigRat_AM is not None:
                    sigRat_AM.push(sig_ratio.unsqueeze(0))

                # ── H1: lag-1 autocorrelation corr(H_t, H_{t-1}) within
                # trajectory. High (>0.5) → signal has temporal structure
                # (not pure noise). Low (<0.2) → signal is instantaneous
                # noise dominated → supports H1 (signal has no exploitable
                # structure). Only computed over (t, t-1) pairs both valid.
                if entropy_list.size(2) >= 2:
                    a = entropy_list[:, :, 1:]                                   # H_t
                    b = entropy_list[:, :, :-1]                                  # H_{t-1}
                    m_pair = (valid[:, :, 1:].float() * valid[:, :, :-1].float())
                    n_pair = m_pair.sum().clamp(min=1.0)
                    mu_a = (a * m_pair).sum() / n_pair
                    mu_b = (b * m_pair).sum() / n_pair
                    cov  = (((a - mu_a) * (b - mu_b)) * m_pair).sum() / n_pair
                    var_a = (((a - mu_a) ** 2) * m_pair).sum() / n_pair
                    var_b = (((b - mu_b) ** 2) * m_pair).sum() / n_pair
                    lag1  = (cov / (var_a.sqrt() * var_b.sqrt()).clamp(min=1e-8)).clamp(min=-1.0, max=1.0)
                    if lagCorr_AM is not None:
                        lagCorr_AM.push(lag1.unsqueeze(0))
        else:
            log_prob = prob_list.log().sum(dim=2)

        loss = -(advantage * log_prob).mean()

        if USE_ENTROPY_BONUS:
            loss = loss - ENTROPY_BONUS_BETA * entropy_list.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # ── Logging ──────────────────────────────────────────────────────
        max_reward, _ = reward.max(dim=1)
        score_AM.push(-max_reward.float())
        loss_AM.push(loss.detach().unsqueeze(0))

        if time.time() - logger_start > LOG_PERIOD_SEC or episode >= TRAIN_EPISODES:
            elapsed = time.strftime("%H:%M:%S", time.gmtime(time.time() - timer_start))
            extra = ""
            phase = "warmup" if not apply_pert else "active"
            if USE_TRAJINTERNAL_BASELINE:
                mode = "Tin"
            elif USE_MONOSEG_BASELINE:
                mode = "Mseg"
            else:
                mode = "Z"
            if rw_AM is not None and rw_AM.count > 0:
                extra = "  {}({}): top3={:.3f} rw={:.3f} ctTop3={:.3f}".format(
                    mode, phase,
                    top3_AM.result() if (top3_AM is not None and top3_AM.count > 0) else float('nan'),
                    rw_AM.result(),
                    ctTop3_AM.result() if (ctTop3_AM is not None and ctTop3_AM.count > 0) else float('nan'))
                if USE_MONOSEG_BASELINE and nsegs_AM is not None and nsegs_AM.count > 0:
                    extra += "  nsegs={:.1f} seglen={:.1f}".format(
                        nsegs_AM.result(), seglen_AM.result())
                # H1-H4 diagnostics
                extra += ("  | H1:MLPR²={:.3f} lag1={:+.3f}"
                          "  H2:ω²={:.3f}"
                          "  H3:ctCV={:.3f}"
                          "  H4:σ²ratio={:.3f}").format(
                    mlpR2_AM.result()  if (mlpR2_AM  is not None and mlpR2_AM.count  > 0) else float('nan'),
                    lagCorr_AM.result()if (lagCorr_AM is not None and lagCorr_AM.count> 0) else float('nan'),
                    omega_AM.result()  if (omega_AM  is not None and omega_AM.count  > 0) else float('nan'),
                    ctCV_AM.result()   if (ctCV_AM   is not None and ctCV_AM.count   > 0) else float('nan'),
                    sigRat_AM.result() if (sigRat_AM is not None and sigRat_AM.count > 0) else float('nan'))
            logger.info('Ep:{:03d}-{:07d}({:5.1f}%)  T:{}  Loss:{:+.4f}  Avg.best_dist:{:.4f}{}'.format(
                epoch, episode, 100. * episode / TRAIN_EPISODES,
                elapsed, loss_AM.result(), score_AM.result(), extra))
            logger_start = time.time()

    lr_scheduler.step()
