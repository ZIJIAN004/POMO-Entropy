"""
Train.py — POMO REINFORCE with optional pure group-wise entropy reweighting.

Supported modes (mutually exclusive):
  • USE_ENTROPY_REWEIGHT = False : baseline POMO (no reweighting)
  • USE_ENTROPY_REWEIGHT = True  : per-step c_t = 1 + γ · sign(A) · ΔH_t
                                   where ΔH is within-group z-score of entropy
                                   over (instance, n_feasible[, at_depot,
                                   load_bin, vis_ratio_bin]).
                                   First ENTROPY_WARMUP_EPOCHS epochs run with
                                   ΔH = 0 (baseline POMO) but still log
                                   monitoring stats.

  • USE_ENTROPY_BONUS  : A2C-style entropy bonus (independent, can stack)

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

    logger_start = time.time()
    episode      = 0
    device       = next(model.parameters()).device

    # True once warmup is done — apply the actual perturbation. During warmup
    # we still compute group stats (for monitoring) but force ΔH = 0.
    apply_pert = USE_ENTROPY_REWEIGHT and (epoch > ENTROPY_WARMUP_EPOCHS)

    if USE_ENTROPY_REWEIGHT and PROBLEM_TYPE == 'vrptw':
        raise NotImplementedError(
            "USE_ENTROPY_REWEIGHT is not supported for vrptw yet; "
            "set it to False in HYPER_PARAMS or pass --mode off.")

    # Collect entropy + group-construction features only when reweighting is on.
    collect_groups = USE_ENTROPY_REWEIGHT or USE_ENTROPY_BONUS
    needs_cvrp_feats = (USE_ENTROPY_REWEIGHT and PROBLEM_TYPE == 'cvrp')

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
        if needs_cvrp_feats:
            at_depot_list  = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
            load_list      = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
            vis_ratio_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if PROBLEM_TYPE == 'cvrp' and USE_ENTROPY_REWEIGHT:
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

            if needs_cvrp_feats:
                if state.current_node is None:
                    at_depot = torch.zeros(batch_size, POMO_SIZE, device=device)
                else:
                    at_depot = (state.current_node == 0).float()
                at_depot_list = torch.cat(
                    (at_depot_list, at_depot[:, :, None]), dim=2)

                load_list = torch.cat(
                    (load_list, state.load[:, :, None]), dim=2)

                vis_step = state.visited_customer_count.float() / max(PROBLEM_SIZE, 1)
                vis_ratio_list = torch.cat(
                    (vis_ratio_list, vis_step[:, :, None]), dim=2)

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

            if PROBLEM_TYPE == 'tsp':
                gid, n_grp = build_group_id(
                    'tsp', n_feasible=n_feasible_list, n_bins=ENTROPY_N_BINS)
            else:
                gid, n_grp = build_group_id(
                    PROBLEM_TYPE,
                    n_feasible=n_feasible_list,
                    at_depot=at_depot_list,
                    load=load_list,
                    vis_ratio=vis_ratio_list,
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
            )
            log_prob = (prob_list.log() * weights).sum(dim=2)

            if top3_AM is not None:
                top3_AM.push(diag['top3_concentration'].unsqueeze(0))
            if small_AM is not None:
                small_AM.push(diag['small_group_ratio'].unsqueeze(0))
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
                extra = "  Z({}):top3={:.3f} small={:.3f}".format(
                    phase, top3_AM.result(), small_AM.result())
            logger.info('Ep:{:03d}-{:07d}({:5.1f}%)  T:{}  Loss:{:+.4f}  Avg.best_dist:{:.4f}{}'.format(
                epoch, episode, 100. * episode / TRAIN_EPISODES,
                elapsed, loss_AM.result(), score_AM.result(), extra))
            logger_start = time.time()

    lr_scheduler.step()
