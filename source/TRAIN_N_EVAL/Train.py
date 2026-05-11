"""
Train.py — POMO REINFORCE with optional entropy schemes.

Supported modes:
  • baseline POMO (no reweighting)
  • USE_ENTROPY_WEIGHT      : original log-F OLS based reweighting
  • USE_ENTROPY_BONUS       : A2C-style entropy bonus
  • USE_MODE_B_BASELINE     : shared MLP + per-instance OLS based reweighting
                              (overrides USE_ENTROPY_WEIGHT when set)

advantage_i = r_i - mean_j(r_j)
"""

import time
import torch
import torch.nn.functional as F

from HYPER_PARAMS import *
from source.utilities import Average_Meter

if USE_ENTROPY_WEIGHT and not USE_MODE_B_BASELINE:
    from source.entropy_utils import compute_entropy_weights

if USE_MODE_B_BASELINE:
    from source.baseline import (
        batched_per_instance_ols,
        compute_mode_b_weights,
        extract_state_features_tsp,
        extract_state_features_cvrp,
        extract_state_features_vrptw,
    )
    if PROBLEM_TYPE == 'tsp':
        _EXTRACT_FEATS = extract_state_features_tsp
    elif PROBLEM_TYPE == 'cvrp':
        _EXTRACT_FEATS = extract_state_features_cvrp
    elif PROBLEM_TYPE == 'vrptw':
        _EXTRACT_FEATS = extract_state_features_vrptw
    else:
        raise ValueError(f"Mode B: unsupported PROBLEM_TYPE={PROBLEM_TYPE}")

_COLLECT_ENTROPY = USE_ENTROPY_WEIGHT or USE_ENTROPY_BONUS or USE_MODE_B_BASELINE


def TRAIN(model, env, optimizer, lr_scheduler, epoch, timer_start, logger,
          baseline_module=None, baseline_optim=None):
    model.train()

    score_AM      = Average_Meter()
    loss_AM       = Average_Meter()
    baseline_AM   = Average_Meter() if baseline_module is not None else None

    logger_start = time.time()
    episode      = 0
    device       = next(model.parameters()).device

    # whether to use mode-B residuals for reweighting (post-warmup)
    use_mode_b_weights = (USE_MODE_B_BASELINE and
                          baseline_module is not None and
                          epoch > MODE_B_WARMUP_EPOCHS)

    while episode < TRAIN_EPISODES:
        batch_size = min(TRAIN_BATCH_SIZE, TRAIN_EPISODES - episode)
        episode   += batch_size

        # ── Rollout ──────────────────────────────────────────────────────
        env.load_problems(batch_size, device=device)
        reset_state, _, _ = env.reset()
        model.pre_forward(reset_state)

        prob_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if _COLLECT_ENTROPY:
            entropy_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if USE_ENTROPY_WEIGHT or USE_MODE_B_BASELINE:
            n_feasible_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)
        if USE_MODE_B_BASELINE:
            state_feat_list = []                       # accumulate per step, stack later
            finished_list   = torch.zeros(batch_size, POMO_SIZE, 0,
                                          dtype=torch.bool, device=device) \
                              if PROBLEM_TYPE in ('cvrp', 'vrptw') else None

        state, reward, done = env.pre_step()

        while not done:
            selected, prob, entropy = model(state)
            if _COLLECT_ENTROPY:
                entropy_list = torch.cat((entropy_list, entropy[:, :, None]), dim=2)
            if USE_ENTROPY_WEIGHT or USE_MODE_B_BASELINE:
                n_feasible_list = torch.cat(
                    (n_feasible_list, state.n_feasible[:, :, None].float()), dim=2)
            if USE_MODE_B_BASELINE:
                state_feat_list.append(_EXTRACT_FEATS(state, PROBLEM_SIZE))   # (B, P, n_state)
                if finished_list is not None:
                    finished_list = torch.cat(
                        (finished_list, state.finished[:, :, None]), dim=2)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # ── REINFORCE advantage ──────────────────────────────────────────
        reward_f  = reward.float()
        advantage = reward_f - reward_f.mean(dim=1, keepdim=True)

        # ── Mode B: train baseline MLP + (optionally) compute residual weights ──
        if USE_MODE_B_BASELINE and baseline_module is not None:
            # (B, P, T, n_state)
            state_feat = torch.stack(state_feat_list, dim=2)
            T_total    = state_feat.size(2)

            # validity mask: skip forced steps + finished padding
            valid = torch.ones(batch_size, POMO_SIZE, T_total,
                               dtype=torch.bool, device=device)
            # steps 0 and 1 are forced (selected_count == 0/1 → entropy = 0)
            forced_steps = 1 if PROBLEM_TYPE == 'tsp' else 2
            if T_total > forced_steps:
                valid[:, :, :forced_steps] = False
            else:
                valid[:] = False
            if finished_list is not None:
                valid &= ~finished_list

            # instance embedding from policy encoder (DETACHED — encoder frozen
            # w.r.t. baseline training)
            inst_emb = model.encoded_nodes.mean(dim=1).detach()       # (B, embed_dim)
            inst_emb_per_step = inst_emb[:, None, None, :].expand(
                batch_size, POMO_SIZE, T_total, inst_emb.size(-1))

            # forward MLP → per-instance closed-form OLS
            features = baseline_module(state_feat, inst_emb_per_step)
            H_hat, _ = batched_per_instance_ols(
                features, entropy_list, valid, ridge=MODE_B_RIDGE)

            # train MLP (always, even during warmup)
            if valid.any():
                loss_b = F.mse_loss(H_hat[valid], entropy_list[valid].detach())
                baseline_optim.zero_grad()
                loss_b.backward()
                baseline_optim.step()
                if baseline_AM is not None:
                    baseline_AM.push(loss_b.detach().unsqueeze(0))

            # post-warmup: recompute residual with frozen MLP for weight calc
            if use_mode_b_weights:
                with torch.no_grad():
                    features_f = baseline_module(state_feat, inst_emb_per_step)
                    H_hat_f, _ = batched_per_instance_ols(
                        features_f, entropy_list, valid, ridge=MODE_B_RIDGE)
                residual = entropy_list - H_hat_f
                weights = compute_mode_b_weights(
                    residual, n_feasible_list, advantage, valid, gamma=ENTROPY_GAMMA)
                log_prob = (prob_list.log() * weights).sum(dim=2)
            else:
                log_prob = prob_list.log().sum(dim=2)

        elif USE_ENTROPY_WEIGHT:
            weights  = compute_entropy_weights(
                entropy_list, n_feasible_list, advantage, ENTROPY_GAMMA)
            log_prob = (prob_list.log() * weights).sum(dim=2)
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
            if baseline_AM is not None and baseline_AM.count > 0:
                tag = "ModeB(use)" if use_mode_b_weights else "ModeB(warmup)"
                extra = "  {}_MSE:{:.4f}".format(tag, baseline_AM.result())
            logger.info('Ep:{:03d}-{:07d}({:5.1f}%)  T:{}  Loss:{:+.4f}  Avg.best_dist:{:.4f}{}'.format(
                epoch, episode, 100. * episode / TRAIN_EPISODES,
                elapsed, loss_AM.result(), score_AM.result(), extra))
            logger_start = time.time()

    lr_scheduler.step()
