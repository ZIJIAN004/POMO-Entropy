"""Evaluate.py — Greedy evaluation"""

import torch
from HYPER_PARAMS import *
from source.utilities import Average_Meter

eval_result = []


def EVAL(model, env, epoch, timer_start, logger, result_folder_path):
    import os
    global eval_result

    model.eval()
    score_AM = Average_Meter()
    device   = next(model.parameters()).device

    with torch.no_grad():
        episode = 0
        while episode < EVAL_EPISODES:
            batch_size = min(TEST_BATCH_SIZE, EVAL_EPISODES - episode)
            episode   += batch_size

            env.load_problems(batch_size, device=device)
            reset_state, _, _ = env.reset()
            model.pre_forward(reset_state)

            state, reward, done = env.pre_step()
            while not done:
                selected, _ = model(state)
                state, reward, done = env.step(selected)

            max_reward, _ = reward.max(dim=1)
            score_AM.push(-max_reward.float())

    avg_dist = score_AM.result()
    eval_result.append(avg_dist)

    if avg_dist == min(eval_result):
        torch.save(model.state_dict(),
                   os.path.join(result_folder_path, 'MODEL_BEST.pt'))

    model.train()
    logger.info('----------------------------------------------------------------------')
    logger.info('  <<< EVAL Ep:{:03d} >>>   Avg.best_dist:{:.5f}'.format(epoch, avg_dist))
    logger.info('eval_result = {}'.format(eval_result))
    logger.info('----------------------------------------------------------------------')
    return avg_dist
