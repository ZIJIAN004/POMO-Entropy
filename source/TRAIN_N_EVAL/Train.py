"""
Train.py — POMO REINFORCE with optional entropy schemes

advantage_i = r_i - mean_j(r_j)
loss = -mean(advantage * (weighted_)log_prob) - beta * mean(entropy)
"""

import time
import torch

from HYPER_PARAMS import *
from source.utilities import Average_Meter

if USE_ENTROPY_WEIGHT:
    from source.entropy_utils import compute_entropy_weights

_COLLECT_ENTROPY = USE_ENTROPY_WEIGHT or USE_ENTROPY_BONUS


def TRAIN(model, env, optimizer, lr_scheduler, epoch, timer_start, logger):
    model.train()

    score_AM = Average_Meter()
    loss_AM  = Average_Meter()

    logger_start = time.time()
    episode      = 0
    device       = next(model.parameters()).device

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
        if USE_ENTROPY_WEIGHT:
            n_feasible_list = torch.zeros(batch_size, POMO_SIZE, 0, device=device)

        state, reward, done = env.pre_step()

        while not done:
            selected, prob, entropy = model(state)
            if _COLLECT_ENTROPY:
                entropy_list = torch.cat((entropy_list, entropy[:, :, None]), dim=2)
            if USE_ENTROPY_WEIGHT:
                n_feasible_list = torch.cat((n_feasible_list, state.n_feasible[:, :, None].float()), dim=2)
            state, reward, done = env.step(selected)
            prob_list = torch.cat((prob_list, prob[:, :, None]), dim=2)

        # ── REINFORCE ────────────────────────────────────────────────────
        reward_f  = reward.float()
        advantage = reward_f - reward_f.mean(dim=1, keepdim=True)

        if USE_ENTROPY_WEIGHT:
            weights  = compute_entropy_weights(
                entropy_list, n_feasible_list, advantage,
                ENTROPY_GAMMA)
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
            logger.info('Ep:{:03d}-{:07d}({:5.1f}%)  T:{}  Loss:{:+.4f}  Avg.best_dist:{:.4f}'.format(
                epoch, episode, 100. * episode / TRAIN_EPISODES,
                elapsed, loss_AM.result(), score_AM.result()))
            logger_start = time.time()

    lr_scheduler.step()
