#!/bin/bash
# ============================================================================
# submit_cvrp_signal_compare_zhuoyi.sh — CVRP100 信号源对照 (2 GPU, 同时跑)
#
# 同一个 sbatch 申请 2 张卡，两个 python 进程并发：
#   GPU 0: --signal entropy   (基线 confidence signal)
#   GPU 1: --signal margin    (prob top1−top2，scale-free)
# 其余参数完全一致：4 维 bucket, softmax, γ=0.3, 200 ep, seed 1
#
# 用法（zhuoyi 登录节点）：
#   cd /homes/zhuoyi/zijianliu/POMO-Entropy
#   sbatch submit_cvrp_signal_compare_zhuoyi.sh
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=2
#SBATCH --job-name=cvrp_signal_cmp
#SBATCH --output=/homes/zhuoyi/zijianliu/POMO-Entropy/logs/cvrp_signal_cmp_%j.log

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

trap 'notify "POMO CVRP100 signal-compare 异常" "job=${JOB} at $(date)"' ERR
trap 'notify "POMO CVRP100 signal-compare 中断" "job=${JOB} at $(date)"' INT TERM

echo "[$(date '+%F %T')] job=${JOB}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

LOG_E="logs/cvrp_entropy_${JOB}.log"
LOG_M="logs/cvrp_margin_${JOB}.log"

notify "POMO CVRP100 signal-compare 启动" \
"job=${JOB} (2 GPU)
GPU 0: --signal entropy  → ${LOG_E}
GPU 1: --signal margin   → ${LOG_M}
共同: cvrp size=100 softmax=on γ=0.3 epoch=200 seed=1
含 OLS R² (F2/F5) 诊断"

# ── GPU 0: entropy 对照 ─────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=0 python -u Train.py \
    --problem cvrp --size 100 --softmax on \
    --signal entropy --gamma 0.3 --epoch 200 --seed 1 \
    > "${LOG_E}" 2>&1 &
PID_E=$!

# ── GPU 1: margin 新信号 ────────────────────────────────────────────────────
CUDA_VISIBLE_DEVICES=1 python -u Train.py \
    --problem cvrp --size 100 --softmax on \
    --signal margin --gamma 0.3 --epoch 200 --seed 1 \
    > "${LOG_M}" 2>&1 &
PID_M=$!

echo "[$(date '+%F %T')] PID_entropy=${PID_E} PID_margin=${PID_M}"

wait ${PID_E};  EXIT_E=$?
wait ${PID_M};  EXIT_M=$?

SUMMARY="entropy exit=${EXIT_E}  log=${LOG_E}
margin  exit=${EXIT_M}  log=${LOG_M}"

if [ "${EXIT_E}" -eq 0 ] && [ "${EXIT_M}" -eq 0 ]; then
    notify "POMO CVRP100 signal-compare 完成 ✓" "${SUMMARY}"
else
    notify "POMO CVRP100 signal-compare 部分失败 ✗" "${SUMMARY}"
fi

exit $(( EXIT_E | EXIT_M ))
