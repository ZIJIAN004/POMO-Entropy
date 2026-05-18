#!/bin/bash
# ============================================================================
# submit_cvrp_diag_zhuoyi.sh — CVRP100 H1-H4 诊断 sbatch, 30 ep, long QOS
#
# 一个 sbatch, 单 GPU：
#   * 训练用 4 维 bucket + softmax reweight (主路径)
#   * 同进程内并行跑 MLP 诊断器（每 batch 收 features+signal, fit 5 步,
#     算 held-out R²）作为 H1 信号 upper bound
#
# 输出 log 行包含 4 个 H1-H4 客观指标：
#   H1: MLPR² (signal upper bound, low → 信号本身无结构)
#       lag1   (autocorrelation, low → 信号是噪声)
#   H2: ω²    (ANOVA effect size, 不依赖 feature 形式)
#   H3: ctCV  (c_t 变异系数, low → softmax 几乎均匀, reweight 无效)
#   H4: σ²ratio (σ²_traj / σ²_step, >1 → 信号是 trajectory-level)
#
# 用法（zhuoyi 登录节点）：
#   cd /homes/zhuoyi/zijianliu/POMO-Entropy
#   sbatch submit_cvrp_diag_zhuoyi.sh
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=1
#SBATCH --job-name=cvrp_diag
#SBATCH --output=/homes/zhuoyi/zijianliu/POMO-Entropy/logs/cvrp_diag_%j.log

export HOME=/homes/zhuoyi
export PIP_CACHE_DIR=/homes/zhuoyi/.pip_cache
export TMPDIR=/homes/zhuoyi/tmp
export XDG_CACHE_HOME=/homes/zhuoyi/.cache
export TRITON_CACHE_DIR=/homes/zhuoyi/.triton

source /homes/zhuoyi/.bashrc
eval "$(conda shell.bash hook)"
conda activate pomo-entropy

cd /homes/zhuoyi/zijianliu/POMO-Entropy
mkdir -p logs

SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
JOB="${SLURM_JOB_ID:-local}"

notify() {
    curl -s -m 10 "https://sctapi.ftqq.com/${SCKEY}.send" \
        --data-urlencode "title=$1" --data-urlencode "desp=$2" > /dev/null 2>&1 || true
}

trap 'notify "POMO CVRP100 H1-H4 诊断 异常" "job=${JOB} exit=$? at $(date)"' ERR
trap 'notify "POMO CVRP100 H1-H4 诊断 中断" "job=${JOB} at $(date)"' INT TERM

echo "[$(date '+%F %T')] job=${JOB}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

notify "POMO CVRP100 H1-H4 诊断 启动" \
"job=${JOB}
problem=cvrp size=100 softmax=on gamma=0.3 epoch=30 seed=1
4 维 bucket 主路径 + MLP 并行诊断
新 log 字段: H1:MLPR² lag1 / H2:ω² / H3:ctCV / H4:σ²ratio"

python -u Train.py --problem cvrp --size 100 --softmax on --gamma 0.3 --epoch 30 --seed 1

EXIT=$?

if [ "${EXIT}" -eq 0 ]; then
    notify "POMO CVRP100 H1-H4 诊断 完成 ✓" "job=${JOB} 正常结束 at $(date)"
else
    notify "POMO CVRP100 H1-H4 诊断 失败 ✗" "job=${JOB} exit=${EXIT} at $(date)"
fi

exit "${EXIT}"
