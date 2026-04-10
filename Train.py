"""
POMO Baseline — 纯 POMO REINFORCE 训练

支持 TSP / CVRP / VRPTW，通过 HYPER_PARAMS.PROBLEM_TYPE 选择。
数据生成逻辑与 UniCOP-Reason 一致。

Run:
    python Train.py
"""

from HYPER_PARAMS import *

SAVE_FOLDER_NAME = "POMO_{}_n{}".format(PROBLEM_TYPE.upper(), PROBLEM_SIZE)
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
plt.show()
