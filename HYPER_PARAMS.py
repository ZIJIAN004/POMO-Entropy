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
ENTROPY_GAMMA        = 1.0         # softmax temperature (larger = sharper)

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
# Mode B Baseline (Learned Entropy Baseline)
#   shared MLP φ_θ produces h-dim features from (state, instance_emb);
#   per-instance β_b is solved via closed-form OLS each batch.
#   residual = H - φ·β_b  → replaces (H - a·logF - b) in z-score pipeline.
#
#   Encoder is detached when feeding into the MLP (`.detach()` on encoded_nodes),
#   so baseline training does NOT propagate gradients back into the policy encoder.
# ===========================================================================
USE_MODE_B_BASELINE    = True       # if True, overrides USE_ENTROPY_WEIGHT
MODE_B_WARMUP_EPOCHS   = 20         # epochs to train MLP without using its output
                                    # (let encoder + MLP stabilize first)
MODE_B_HIDDEN          = 16         # MLP hidden width
MODE_B_LR              = 1e-3       # 10x policy LR — fast adapt to encoder changes
MODE_B_WEIGHT_DECAY    = 1e-3
MODE_B_RIDGE           = 1e-4       # ridge regularization for OLS

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
