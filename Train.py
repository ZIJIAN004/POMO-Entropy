"""
POMO — REINFORCE with optional pure group-wise entropy reweighting.

Run:
    python Train.py                            # use HYPER_PARAMS defaults
    python Train.py --problem tsp              # override problem type
    python Train.py --problem cvrp --size 50   # override problem/size
    python Train.py --problem tsp --mode off   # baseline POMO, no reweight
    python Train.py --problem tsp --mode on --gamma 0.3 --warmup 10
"""

import argparse

_parser = argparse.ArgumentParser()
_parser.add_argument('--problem', type=str, default=None,
                     choices=['tsp', 'cvrp', 'vrptw'])
_parser.add_argument('--size', type=int, default=None)
_parser.add_argument('--mode', type=str, default=None,
                     choices=['off', 'on'],
                     help='off: baseline POMO; on: enable entropy reweighting.')
_parser.add_argument('--epoch', type=int, default=None,
                     help='override TOTAL_EPOCH.')
_parser.add_argument('--warmup', type=int, default=None,
                     help='Epochs to delay perturbation (only monitoring during warmup).')
_parser.add_argument('--gamma', type=float, default=None,
                     help='perturbation amplitude for c_t = 1+γ·sign(A)·ΔH_t.')
_parser.add_argument('--bidir', type=str, default=None, choices=['on', 'off'],
                     help='bidirectional normalization: subtract per-trajectory '
                          'mean before bucket z-score.')
_parser.add_argument('--softmax', type=str, default=None, choices=['on', 'off'],
                     help='softmax c_t = softmax(γ·sign(A)·ΔH, dim=step)·T_valid '
                          'instead of linear c_t = 1+γ·sign(A)·ΔH.')
_parser.add_argument('--vis-ratio', type=str, default=None, choices=['on', 'off'],
                     help='include vis_ratio as a binning dim on CVRP/VRPTW. '
                          'off → 3-dim bucket (n_feasible, at_depot, load/time).')
_parser.add_argument('--lowg-thresh', type=float, default=None,
                     help='strip rw eligibility from buckets with grp_mean < '
                          'this threshold. 0 = no filter; 0.05 ≈ "low-entropy mask".')
_parser.add_argument('--monoseg', type=str, default=None, choices=['on', 'off'],
                     help='use monotonic-segment baseline instead of bucket: '
                          'ΔH_t = H_t − H[last_reversal_t], softmax over trajectory.')
_parser.add_argument('--postbucket', type=str, default=None, choices=['on', 'off'],
                     help='only with --monoseg on: subtract bucket-mean of '
                          'ΔH_local within (n_feasible, at_depot, load_bin).')
_parser.add_argument('--robust', type=str, default=None, choices=['on', 'off'],
                     help='use median+IQR/1.349 instead of mean+std for bucket '
                          'normalization. Outlier-robust on small buckets.')
_parser.add_argument('--seed', type=int, default=None,
                     help='random seed for torch/numpy/random (None = no seeding, '
                          'CUDA non-deterministic). Run name gets -s{seed} suffix.')
_args, _ = _parser.parse_known_args()

import HYPER_PARAMS as _HP
if _args.problem is not None:
    _HP.PROBLEM_TYPE = _args.problem
if _args.size is not None:
    _HP.PROBLEM_SIZE = _args.size
    _HP.POMO_SIZE    = _args.size
if _args.mode is not None:
    _HP.USE_ENTROPY_REWEIGHT = (_args.mode == 'on')
if _args.epoch is not None:
    _HP.TOTAL_EPOCH = _args.epoch
if _args.warmup is not None:
    _HP.ENTROPY_WARMUP_EPOCHS = _args.warmup
if _args.gamma is not None:
    _HP.ENTROPY_GAMMA = _args.gamma
if _args.bidir is not None:
    _HP.USE_BIDIR_NORM = (_args.bidir == 'on')
if _args.softmax is not None:
    _HP.USE_SOFTMAX_NORM = (_args.softmax == 'on')
if _args.vis_ratio is not None:
    _HP.USE_VIS_RATIO_BIN = (_args.vis_ratio == 'on')
if _args.lowg_thresh is not None:
    _HP.LOW_GRP_MEAN_THRESH = _args.lowg_thresh
if _args.monoseg is not None:
    _HP.USE_MONOSEG_BASELINE = (_args.monoseg == 'on')
if _args.postbucket is not None:
    _HP.USE_MONOSEG_POSTBUCKET = (_args.postbucket == 'on')
if _args.robust is not None:
    _HP.USE_ROBUST_NORM = (_args.robust == 'on')

from HYPER_PARAMS import *

_tag = ""
if USE_ENTROPY_REWEIGHT:
    _tag += "-Z_g{}_w{}".format(ENTROPY_GAMMA, ENTROPY_WARMUP_EPOCHS)
    if USE_MONOSEG_BASELINE:
        _tag += "-Mseg"
        if USE_MONOSEG_POSTBUCKET:
            _tag += "-PB"
    else:
        if USE_BIDIR_NORM:
            _tag += "-Bd"
        if USE_SOFTMAX_NORM:
            _tag += "-Sm"
        if PROBLEM_TYPE in ('cvrp', 'vrptw') and not USE_VIS_RATIO_BIN:
            _tag += "-noVR"
        if LOW_GRP_MEAN_THRESH > 0.0:
            _tag += "-Lm{}".format(LOW_GRP_MEAN_THRESH)
        if USE_ROBUST_NORM:
            _tag += "-RN"
if USE_ENTROPY_BONUS:
    _tag += "-Bonus_b{}".format(ENTROPY_BONUS_BETA)
if _args.seed is not None:
    _tag += "-s{}".format(_args.seed)
SAVE_FOLDER_NAME = "POMO_{}_n{}{}".format(PROBLEM_TYPE.upper(), PROBLEM_SIZE, _tag)
print(SAVE_FOLDER_NAME)

import os
import random
import shutil
import time
import numpy as np
import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_sched

# ── Seed control ─────────────────────────────────────────────────────────────
if _args.seed is not None:
    random.seed(_args.seed)
    np.random.seed(_args.seed)
    torch.manual_seed(_args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(_args.seed)
    print(f"[seed] random/numpy/torch seeded with {_args.seed}")
import matplotlib
matplotlib.use('Agg')   # headless-safe: no display, no blocking plt.show()
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
    # 从目录名解析 epoch：CheckPoint_ep00050 → 50
    ckpt_dir_name = os.path.basename(RESUME_CKPT_PATH.rstrip('/\\'))
    start_epoch = int(ckpt_dir_name.split('ep')[-1]) + 1
    logger.info('Resumed from {} (start_epoch={})'.format(RESUME_CKPT_PATH, start_epoch))

# ── Training loop ─────────────────────────────────────────────────────────────
timer_start       = time.time()
checkpoint_epochs = set(np.arange(1, TOTAL_EPOCH + 1, MODEL_SAVE_INTERVAL).tolist())

for epoch in range(start_epoch, TOTAL_EPOCH + 1):

    TRAIN(model, env, optimizer, lr_scheduler,
          epoch=epoch, timer_start=timer_start, logger=logger)
    EVAL(model, env, epoch=epoch, timer_start=timer_start,
         logger=logger, result_folder_path=result_folder_path)

    if epoch in checkpoint_epochs:
        ckpt_path = os.path.join(result_folder_path, 'CheckPoint_ep{:05d}'.format(epoch))
        os.makedirs(ckpt_path, exist_ok=True)
        torch.save(model.state_dict(), os.path.join(ckpt_path, 'MODEL_state_dic.pt'))
        torch.save(optimizer.state_dict(), os.path.join(ckpt_path, 'OPTIM_state_dic.pt'))
        torch.save(lr_scheduler.state_dict(), os.path.join(ckpt_path, 'LRSTEP_state_dic.pt'))

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
plt.close()
