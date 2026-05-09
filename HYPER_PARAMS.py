"""
POMO Baseline — 多问题支持（TSP / CVRP / VRPTW）

通过 PROBLEM_TYPE 选择问题类型，数据生成逻辑与 UniCOP-Reason 一致。
"""

# ===========================================================================
# Problem
# ===========================================================================
PROBLEM_TYPE = 'cvrp'    # 'tsp' | 'cvrp' | 'vrptw'
PROBLEM_SIZE = 100
POMO_SIZE    = 100       # = PROBLEM_SIZE

# ===========================================================================
# Training
# ===========================================================================
TOTAL_EPOCH        = 400
TRAIN_EPISODES     = 100 * 1000    # 每 epoch 训练 100k 实例
EVAL_EPISODES      = 10 * 1000     # 每 epoch 评估 10k 实例
TRAIN_BATCH_SIZE   = 64
TEST_BATCH_SIZE    = 256

# ===========================================================================
# Model Architecture
# ===========================================================================
EMBEDDING_DIM     = 128
ENCODER_LAYER_NUM = 6
QKV_DIM           = 16
HEAD_NUM          = 8
FF_HIDDEN_DIM     = 512
LOGIT_CLIPPING    = 10

# ===========================================================================
# Optimization
# ===========================================================================
LEARNING_RATE  = 1e-4
WEIGHT_DECAY   = 1e-6
LR_MILESTONES  = [381, 391]
LR_GAMMA       = 0.1

# ===========================================================================
# Entropy-Weighted Advantage Modulation (scheme: per-step reweight)
# ===========================================================================
USE_ENTROPY_WEIGHT   = True        # False = disable per-step reweight
ENTROPY_GAMMA        = 0.1         # softmax temperature (larger = sharper)

# ===========================================================================
# Entropy Regularization Bonus (scheme A: standard A2C/PPO-style)
#   loss = policy_loss - ENTROPY_BONUS_BETA * mean(entropy)
#   beta > 0  -> encourage exploration (higher entropy)
#   beta < 0  -> encourage commitment  (lower entropy)
# Independent of USE_ENTROPY_WEIGHT; can be combined or used alone.
# ===========================================================================
USE_ENTROPY_BONUS    = False
ENTROPY_BONUS_BETA   = 0.01

# ===========================================================================
# Checkpoint & Resume
# ===========================================================================
RESUME                = False     # 是否从 checkpoint 断点续训
RESUME_CKPT_PATH      = ''       # checkpoint 目录路径，例如 'result/POMO_CVRP_n100/.../CheckPoint_ep00050'

# ===========================================================================
# Logging
# ===========================================================================
LOG_PERIOD_SEC      = 30
MODEL_SAVE_INTERVAL = 50
