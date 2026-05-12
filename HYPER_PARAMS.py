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
# Entropy-Weighted Advantage Modulation
# Two feature schemes for the per-instance OLS baseline:
#   • USE_HAND_FEATURES : hand-crafted features go directly into OLS
#                         (TSP: [log F, 1]; CVRP/VRPTW: [log F, load|time,
#                          at_depot, visited_customer_ratio, 1])
#   • USE_MLP_FEATURES  : end-to-end MLP baseline. Inputs are encoder-side
#                         embeddings (inst_summary mean-pool, enc_last,
#                         optionally enc_first) plus raw env scalars (F, sc,
#                         load|current_time, current_xy, vis_count). NO
#                         hand-engineered nonlinearities and NO decoder-side
#                         tensors (no q / mh_out — those carry policy
#                         selection-confidence and would absorb the signal
#                         we want as OLS residual). All encoder outputs are
#                         detached so baseline grads do NOT touch the policy.
# When USE_MLP_FEATURES is True it overrides USE_HAND_FEATURES.
# ===========================================================================
USE_HAND_FEATURES    = True        # Mode "hand" — direct OLS on hand-crafted feats
USE_MLP_FEATURES     = True        # Mode "mlp"  — MLP-learned feats + OLS head
ENTROPY_GAMMA        = 1.0         # softmax temperature (larger = sharper)

# ===========================================================================
# Entropy Regularization Bonus (A2C/PPO-style — independent of the OLS path)
#   loss = policy_loss - ENTROPY_BONUS_BETA * mean(entropy)
#   beta > 0  -> encourage exploration (higher entropy)
#   beta < 0  -> encourage commitment  (lower entropy)
# Can be combined with either Hand/MLP mode or used alone.
# ===========================================================================
USE_ENTROPY_BONUS    = False
ENTROPY_BONUS_BETA   = 0.01

# ===========================================================================
# MLP-features mode hyperparameters (only used when USE_MLP_FEATURES = True)
#
# Input = [inst_summary (D), enc_last (D), (enc_first (D) if TSP), raw_scalar (k)]
#   TSP : 3D + 4 = 388 (with EMBEDDING_DIM=128)
#   CVRP: 2D + 6 = 262
#   VRPTW:2D + 6 = 262
# Architecture: Linear(n_in, hidden) → ReLU → Linear(hidden, hidden) → ReLU →
#               Linear(hidden, h_out)
# Per-instance closed-form OLS solved on top of φ ∈ R^h_out each batch.
# ===========================================================================
MLP_WARMUP_EPOCHS    = 20          # epochs to train MLP without using its output
                                   # (let encoder + MLP stabilize first)
MLP_HIDDEN           = 64          # MLP hidden width (2 hidden layers, each this wide)
MLP_H_OUT            = 8           # per-instance OLS feature dim (β has this many entries)
MLP_LR               = 3e-4        # ~3x policy LR (1e-4) — track encoder a bit faster
MLP_WEIGHT_DECAY     = 1e-3
MLP_RIDGE            = 1e-4        # ridge regularization for per-instance OLS

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
