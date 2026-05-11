"""
POMO — REINFORCE with optional entropy-weighted advantage modulation

Run:
    python Train.py                          # use HYPER_PARAMS defaults
    python Train.py --problem tsp            # override problem type
    python Train.py --problem cvrp --size 50 # override problem type and size
"""

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument('--problem', type=str, default=None)
_parser.add_argument('--size', type=int, default=None)
_args, _ = _parser.parse_known_args()

import HYPER_PARAMS as _HP
if _args.problem is not None:
    _HP.PROBLEM_TYPE = _args.problem
if _args.size is not None:
    _HP.PROBLEM_SIZE = _args.size
    _HP.POMO_SIZE    = _args.size

from HYPER_PARAMS import *

_tag = ""
if USE_MODE_B_BASELINE:
    _tag += "-ModeB_g{}_w{}".format(ENTROPY_GAMMA, MODE_B_WARMUP_EPOCHS)
elif USE_ENTROPY_WEIGHT:
    _tag += "-Entropy_g{}".format(ENTROPY_GAMMA)
if USE_ENTROPY_BONUS:
    _tag += "-Bonus_b{}".format(ENTROPY_BONUS_BETA)
SAVE_FOLDER_NAME = "POMO_{}_n{}{}".format(PROBLEM_TYPE.upper(), PROBLEM_SIZE, _tag)
print(SAVE_FOLDER_NAME)

import os
import shutil
import time
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched
from matplotlib import pyplot as plt

from source.utilities import Get_Logger, Extract_from_LogFile
from source.TRAIN_N_EVAL.Train    import TRAIN
from source.TRAIN_N_EVAL.Evaluate import EVAL

# ── 根据问题类型选择 Model 和 Env ─────────────────────────────────────────────
if PROBLEM_TYPE == 'tsp':
    from source.models.tsp_model import TSPModel as Model
    from source.envs.tsp_env     import TSPEnv   as Env
elif PROBLEM_TYPE == 'cvrp':
    from source.models.cvrp_model import CVRPModel as Model
    from source.envs.cvrp_env     import CVRPEnv   as Env
elif PROBLEM_TYPE == 'vrptw':
    from source.models.vrptw_model import VRPTWModel as Model
    from source.envs.vrptw_env     import VRPTWEnv   as Env
else:
    raise ValueError(f"Unknown PROBLEM_TYPE: {PROBLEM_TYPE}")

# ── Setup ─────────────────────────────────────────────────────────────────────
logger, result_folder_path = Get_Logger(SAVE_FOLDER_NAME)
shutil.copy('./HYPER_PARAMS.py',
            os.path.join(result_folder_path, 'used_HYPER_PARAMS.txt'))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ── Model ─────────────────────────────────────────────────────────────────────
model = Model(
    embedding_dim     = EMBEDDING_DIM,
    encoder_layer_num = ENCODER_LAYER_NUM,
    head_num          = HEAD_NUM,
    qkv_dim           = QKV_DIM,
    ff_hidden_dim     = FF_HIDDEN_DIM,
    logit_clipping    = LOGIT_CLIPPING,
).to(device)

optimizer    = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
lr_scheduler = lr_sched.MultiStepLR(optimizer, milestones=LR_MILESTONES, gamma=LR_GAMMA)

# ── Mode B baseline (optional) ───────────────────────────────────────────────
baseline_module = None
baseline_optim  = None
if USE_MODE_B_BASELINE:
    from source.baseline import EntropyBaselineMLP
    if PROBLEM_TYPE == 'tsp':
        n_state, h_out = 4, 4
    elif PROBLEM_TYPE in ('cvrp', 'vrptw'):
        n_state, h_out = 8, 8
    else:
        raise ValueError(f"Mode B not configured for PROBLEM_TYPE={PROBLEM_TYPE}")
    baseline_module = EntropyBaselineMLP(
        n_state = n_state,
        n_inst  = EMBEDDING_DIM,
        hidden  = MODE_B_HIDDEN,
        h_out   = h_out,
    ).to(device)
    baseline_optim = optim.Adam(
        baseline_module.parameters(),
        lr=MODE_B_LR, weight_decay=MODE_B_WEIGHT_DECAY)

env = Env(problem_size=PROBLEM_SIZE, pomo_size=POMO_SIZE)

# ── 断点续训 ──────────────────────────────────────────────────────────────────
start_epoch = 1
if RESUME:
    assert os.path.isdir(RESUME_CKPT_PATH), f"Checkpoint 目录不存在: {RESUME_CKPT_PATH}"
    model.load_state_dict(torch.load(os.path.join(RESUME_CKPT_PATH, 'MODEL_state_dic.pt'),
                                     map_location=device))
    optimizer.load_state_dict(torch.load(os.path.join(RESUME_CKPT_PATH, 'OPTIM_state_dic.pt'),
                                         map_location=device))
    lr_scheduler.load_state_dict(torch.load(os.path.join(RESUME_CKPT_PATH, 'LRSTEP_state_dic.pt'),
                                             map_location=device))
    if baseline_module is not None:
        bp = os.path.join(RESUME_CKPT_PATH, 'BASELINE_state_dic.pt')
        bo = os.path.join(RESUME_CKPT_PATH, 'BASELINE_OPTIM_state_dic.pt')
        if os.path.isfile(bp) and os.path.isfile(bo):
            baseline_module.load_state_dict(torch.load(bp, map_location=device))
            baseline_optim.load_state_dict(torch.load(bo, map_location=device))
            logger.info('Resumed baseline from {}'.format(bp))
        else:
            logger.info('No baseline ckpt found in {}; starting fresh baseline.'
                        .format(RESUME_CKPT_PATH))
    # 从目录名解析 epoch：CheckPoint_ep00050 → 50
    ckpt_dir_name = os.path.basename(RESUME_CKPT_PATH.rstrip('/\\'))
    start_epoch = int(ckpt_dir_name.split('ep')[-1]) + 1
    logger.info('Resumed from {} (start_epoch={})'.format(RESUME_CKPT_PATH, start_epoch))

# ── Training loop ─────────────────────────────────────────────────────────────
timer_start       = time.time()
checkpoint_epochs = set(np.arange(1, TOTAL_EPOCH + 1, MODEL_SAVE_INTERVAL).tolist())

for epoch in range(start_epoch, TOTAL_EPOCH + 1):

    TRAIN(model, env, optimizer, lr_scheduler,
          epoch=epoch, timer_start=timer_start, logger=logger,
          baseline_module=baseline_module, baseline_optim=baseline_optim)
    EVAL(model, env, epoch=epoch, timer_start=timer_start,
         logger=logger, result_folder_path=result_folder_path)

    if epoch in checkpoint_epochs:
        ckpt_path = os.path.join(result_folder_path, 'CheckPoint_ep{:05d}'.format(epoch))
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'MODEL_state_dic.pt'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, 'OPTIM_state_dic.pt'))
        torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_path, 'LRSTEP_state_dic.pt'))
        if baseline_module is not None:
            torch.save(baseline_module.state_dict(),
                       os.path.join(ckpt_path, 'BASELINE_state_dic.pt'))
            torch.save(baseline_optim.state_dict(),
                       os.path.join(ckpt_path, 'BASELINE_OPTIM_state_dic.pt'))

torch.save(model.state_dict(), os.path.join(result_folder_path, 'MODEL_FINAL.pt'))
logger.info('Training complete.')

# ── Plot ──────────────────────────────────────────────────────────────────────
exec_command_str = Extract_from_LogFile(result_folder_path, 'eval_result')
exec(exec_command_str)

plt.figure(figsize=(10, 4))
plt.plot(eval_result, marker='o', markersize=3, linewidth=1)
plt.xlabel('Epoch')
plt.ylabel('Avg. best tour distance')
plt.title('POMO {} – N={}'.format(PROBLEM_TYPE.upper(), PROBLEM_SIZE))
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(result_folder_path, 'eval_result.jpg'))
plt.show()
