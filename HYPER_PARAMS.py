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
# Entropy-weighted advantage modulation (pure group-wise z-score)
#
# Pipeline:
#   1. Partition every (instance) into groups by discrete env features:
#        TSP : (n_feasible,)
#        CVRP: (n_feasible, at_depot, load_bin, vis_ratio_bin)
#               — load and vis_ratio are continuous in [0,1], each split into
#                 ENTROPY_N_BINS equal-width bins.
#   2. Within each group, compute mean/std of entropy.
#        - Groups with < ENTROPY_MIN_GROUP_SIZE valid steps: ΔH = 0 (no perturb)
#   3. ΔH_t = (entropy_t - grp_mean) / grp_std    — per-step "confidence" signal
#                                                   stripped of env-driven entropy.
#   4. Per-step perturbation: c_t = 1 + γ · sign(advantage) · ΔH_t
#                                   (advantage<0 + low-ΔH ⇒ heavy punish on
#                                    confident-but-wrong steps; high-ΔH errors
#                                    forgiven because they're still exploring)
#   5. Warmup: first ENTROPY_WARMUP_EPOCHS epochs run baseline POMO (c_t = 1)
#      but still compute monitoring stats (top3 group concentration etc.).
#
# Monitoring (logged each period):
#   • top3_concentration : per-instance, sum(top-3 group sizes) / total valid steps.
#                          Expected to rise as policy converges (state homogenization).
#   • small_group_ratio  : per-instance, fraction of valid steps falling in
#                          undersized (<min_size) groups. Diagnostic for binning.
# ===========================================================================
USE_ENTROPY_REWEIGHT     = True         # master switch: True = enable reweighting
ENTROPY_GAMMA            = 0.3          # perturbation amplitude for c_t = 1+γ·sign(A)·ΔH_t
ENTROPY_N_BINS           = 10           # equal-width bins per continuous feature
ENTROPY_MIN_GROUP_SIZE   = 4            # groups smaller than this: ΔH = 0
ENTROPY_WARMUP_EPOCHS    = 10           # epochs where ΔH is forced to 0 (monitoring only)

# ── Ablation switches (only effective when USE_ENTROPY_REWEIGHT = True) ─────
# USE_BIDIR_NORM    : subtract per-trajectory mean BEFORE the (instance, gid)
#                     z-score. Diagnostic-motivated: within-bucket variance is
#                     dominated by per-trajectory entropy offset α_i; removing
#                     it makes the z-score's homoscedasticity assumption hold.
# USE_SOFTMAX_NORM  : rw_mask = valid & sufficient_bucket & (n_feasible > 1).
#                     rw steps  : c_t = softmax(γ·sign(A)·ΔH over rw only)·T_rw
#                     valid non-rw (small-group / forced): c_t = 1 (baseline,
#                                   isolated from softmax denominator)
#                     invalid   : c_t = 0
#                     Guarantees c_t ≥ 0 and Σ_t c_t = T_valid; matches the
#                     linear form's c_t=1 behavior on small-group / forced
#                     steps. Without this, linear c_t = 1 + γ·sign(A)·ΔH.
# 4 combinations form the ablation grid. Default (False, False) = current
# linear baseline.
USE_BIDIR_NORM           = False
USE_SOFTMAX_NORM         = False

# ── Group-construction / filter ablations ───────────────────────────────────
# USE_VIS_RATIO_BIN    : include vis_ratio as a binning dimension on CVRP/VRPTW.
#                        Off → bucket = (n_feasible, at_depot, load/time_bin),
#                        3 dims instead of 4 → fewer slots, more steps per bucket,
#                        more stable grp_std. Diagnostic motivation: top3=0.022
#                        observed under 4 dims means avg bucket is tiny.
# LOW_GRP_MEAN_THRESH  : drop rw eligibility from buckets whose grp_mean <
#                        this threshold. 0 = no filter. ~0.05 strips ≈8% of rw
#                        steps that live in near-deterministic buckets where
#                        ΔH ≈ noise (lowG(<.05) diagnostic).
USE_VIS_RATIO_BIN        = True
LOW_GRP_MEAN_THRESH      = 0.0

# ── Monotonic-segment baseline (alternative to bucket baseline) ─────────────
# When True, ΔH_t is computed as H_t − H[anchor_t], where anchor_t is the
# most recent "local extremum" (trend reversal) at or before t in the same
# trajectory. No bucketing, no environment estimation. Softmax over the full
# trajectory's rw steps then redistributes c_t with sign(A)·ΔH_local as the
# logit. Use to test whether trajectory-internal trend signal beats
# bucket-based environment baselines on CVRP/VRPTW.
# When True, USE_VIS_RATIO_BIN / LOW_GRP_MEAN_THRESH / USE_BIDIR_NORM are
# ignored (they apply only to bucket baseline). USE_SOFTMAX_NORM is also
# implicit-True (monoseg always uses softmax).
USE_MONOSEG_BASELINE     = False

# When USE_MONOSEG_BASELINE=True, additionally bucket-normalize ΔH_local by
# state class: subtract the cohort-average ΔH_local from each step's
# ΔH_local. Bucket is always 3-dim (n_feasible, at_depot, load/time_bin) —
# no vis_ratio. Leaves only the trajectory-specific deviation from the
# typical "anchor→t" trend at that state class.
USE_MONOSEG_POSTBUCKET   = False

# Robust normalization: replace mean+std with median+IQR/1.349 per bucket.
# Small buckets with under-estimated grp_std produce |ΔH| >> 1 outliers under
# z-score because std is corrupted by 1-2 extreme values. Quantile-based
# location (median) and scale (IQR) are robust to up to ~25% outliers per
# bucket. Only effective on the bucket path (not monoseg). Default False.
USE_ROBUST_NORM          = False

# ===========================================================================
# Entropy Regularization Bonus (A2C/PPO-style — independent path)
#   loss = policy_loss - ENTROPY_BONUS_BETA * mean(entropy)
#   beta > 0  -> encourage exploration (higher entropy)
#   beta < 0  -> encourage commitment  (lower entropy)
# Can stack on top of the reweighting path or be used alone.
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
