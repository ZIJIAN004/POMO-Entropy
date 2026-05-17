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
from source.baseline import build_group_id, compute_entropy_z_weights


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
    top3_AM  = Average_Meter() if USE_ENTROPY_REWEIGHT else None
    small_AM = Average_Meter() if USE_ENTROPY_REWEIGHT else None
    rw_AM    = Average_Meter() if USE_ENTROPY_REWEIGHT else None
    r2_AM    = Average_Meter() if USE_ENTROPY_REWEIGHT else None   # bucket R² — partition quality
    # H1 diagnostic: corr(ΔH, A) on rw steps, partitioned by GROUP grp_mean.
    # Reweighting operates per-group (grp_mean/grp_std shared by all steps in
    # a bucket), so the right partitioning unit is the group, not the step.
    # Hypothesis: if low-grp_mean groups carry misallocated credit, corr in
    # low-grp half ≪ corr in high-grp half.
    corrH_AM   = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # rw steps in high-grp_mean groups
    corrL_AM   = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # rw steps in low-grp_mean groups
    asnr_AM    = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # |A|/σ_A — H2 probe
    gmMed_AM   = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # median grp_mean among rw steps
    lowGrp_AM  = Average_Meter() if USE_ENTROPY_REWEIGHT else None  # rw steps in groups with grp_mean<0.05

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
            selected, prob, entropy = model(state)

            if collect_groups:
                entropy_list = torch.cat(
                    (entropy_list, entropy[:, :, None]), dim=2)
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

            # USE_VIS_RATIO_BIN=False → 3-dim bucket (drop vis_ratio).
            vis_arg = vis_ratio_list if USE_VIS_RATIO_BIN else None
            if PROBLEM_TYPE == 'tsp':
                gid, n_grp = build_group_id(
                    'tsp', n_feasible=n_feasible_list, n_bins=ENTROPY_N_BINS)
            elif PROBLEM_TYPE == 'cvrp':
                gid, n_grp = build_group_id(
                    'cvrp',
                    n_feasible=n_feasible_list,
                    at_depot=at_depot_list,
                    load=load_list,
                    vis_ratio=vis_arg,
                    n_bins=ENTROPY_N_BINS,
                )
            else:  # vrptw
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
            )
            log_prob = (prob_list.log() * weights).sum(dim=2)

            if top3_AM is not None:
                top3_AM.push(diag['top3_concentration'].unsqueeze(0))
            if small_AM is not None:
                small_AM.push(diag['small_group_ratio'].unsqueeze(0))
            if rw_AM is not None:
                rw_AM.push(diag['rw_ratio'].unsqueeze(0))
            if r2_AM is not None:
                r2_AM.push(diag['r2_grp'].unsqueeze(0))

            # ── H1 diagnostic (TSP-vs-CVRP comparison) ─────────────────────
            # Partition rw steps by their GROUP's grp_mean (median split).
            # Each (instance, gid) bucket shares one grp_mean/grp_std — that's
            # the natural unit of "is this group informative or noise?".
            with torch.no_grad():
                dH    = diag['delta_H']             # (B, P, T)
                rwm   = diag['rw_mask']             # (B, P, T) bool
                gm    = diag['grp_mean']            # (B, P, T), per-step bucket-broadcast
                A_exp = advantage.unsqueeze(2).expand_as(dH)

                gm_rw = gm[rwm]
                if gm_rw.numel() > 100:
                    gm_thresh = gm_rw.median()
                    high = rwm & (gm >  gm_thresh)   # rw steps in high-grp_mean buckets
                    low  = rwm & (gm <= gm_thresh)   # rw steps in low-grp_mean buckets

                    def _pearson(mask):
                        n = int(mask.sum())
                        if n < 100: return None
                        x = dH[mask]
                        y = A_exp[mask]
                        x = x - x.mean(); y = y - y.mean()
                        d = (x.std() * y.std()).clamp(min=1e-8)
                        return ((x * y).mean() / d).unsqueeze(0)

                    c_hi = _pearson(high)
                    c_lo = _pearson(low)
                    if c_hi is not None: corrH_AM.push(c_hi)
                    if c_lo is not None: corrL_AM.push(c_lo)

                    gmMed_AM.push(gm_thresh.unsqueeze(0))
                    # Share of rw steps that live in "near-forced" low-entropy buckets
                    lowGrp = (rwm & (gm < 0.05)).float().sum() / rwm.float().sum().clamp(min=1.0)
                    lowGrp_AM.push(lowGrp.unsqueeze(0))

                A_snr = (advantage.abs() /
                         advantage.std(dim=1, keepdim=True).clamp(min=1e-6)).mean()
                asnr_AM.push(A_snr.unsqueeze(0))
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
            if top3_AM is not None and top3_AM.count > 0:
                phase = "warmup" if not apply_pert else "active"
                extra = "  Z({}):top3={:.3f} small={:.3f} rw={:.3f} R²={:.3f}".format(
                    phase, top3_AM.result(), small_AM.result(), rw_AM.result(),
                    r2_AM.result() if r2_AM.count > 0 else float('nan'))
                # H1 diagnostic readout (group-level partition)
                if corrH_AM is not None and corrH_AM.count > 0:
                    cH = corrH_AM.result()
                    cL = corrL_AM.result() if corrL_AM.count > 0 else float('nan')
                    gmMed = gmMed_AM.result() if gmMed_AM.count > 0 else float('nan')
                    lowG = lowGrp_AM.result() if lowGrp_AM.count > 0 else float('nan')
                    asnr = asnr_AM.result() if asnr_AM.count > 0 else float('nan')
                    extra += "  H1: corr(hiG/loG)={:+.4f}/{:+.4f}  gm_med={:.3f}  lowG(<.05)={:.2f}  |A|/σ_A={:.3f}".format(
                        cH, cL, gmMed, lowG, asnr)
            logger.info('Ep:{:03d}-{:07d}({:5.1f}%)  T:{}  Loss:{:+.4f}  Avg.best_dist:{:.4f}{}'.format(
                epoch, episode, 100. * episode / TRAIN_EPISODES,
                elapsed, loss_AM.result(), score_AM.result(), extra))
            logger_start = time.time()

    lr_scheduler.step()
