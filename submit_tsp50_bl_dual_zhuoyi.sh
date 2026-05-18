#!/bin/bash
# ============================================================================
# submit_tsp50_bl_dual_zhuoyi.sh — TSP50 baseline × 2 seeds, same GPU, 100 ep
#
# 同一张 GPU 上后台并发跑 seed 1 和 seed 2 两个 baseline (mode off)。
# A5000 24GB 对 TSP50 + POMO=50 同时跑两个完全够（每进程 ~5-8GB）。
#
# 用法（zhuoyi 登录节点）：
#   cd /homes/zhuoyi/zijianliu/POMO-Entropy
#   sbatch submit_tsp50_bl_dual_zhuoyi.sh
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=1
#SBATCH --job-name=tsp50_bl_dual
#SBATCH --output=/homes/zhuoyi/zijianliu/POMO-Entropy/logs/tsp50_bl_dual_%j.log

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

trap 'notify "POMO TSP50 baseline dual 异常" "job=${JOB} at $(date)"' ERR
trap 'notify "POMO TSP50 baseline dual 中断" "job=${JOB} at $(date)"' INT TERM

echo "[$(date '+%F %T')] job=${JOB}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

notify "POMO TSP50 baseline dual 启动" \
"job=${JOB}
两个 seed 同卡并发：seed=1, seed=2
problem=tsp size=50 mode=off epoch=100"

LOG1="logs/tsp50_bl_s1_${JOB}.log"
LOG2="logs/tsp50_bl_s2_${JOB}.log"

# 同 GPU 后台启动两个 python 进程
python -u Train.py --problem tsp --size 50 --mode off --epoch 100 --seed 1 \
    > "${LOG1}" 2>&1 &
PID1=$!

python -u Train.py --problem tsp --size 50 --mode off --epoch 100 --seed 2 \
    > "${LOG2}" 2>&1 &
PID2=$!

echo "[$(date '+%F %T')] PID1=${PID1} PID2=${PID2}"

# 等两个都结束
wait ${PID1};  EXIT1=$?
wait ${PID2};  EXIT2=$?

SUMMARY="seed1 exit=${EXIT1} log=${LOG1}
seed2 exit=${EXIT2} log=${LOG2}"

if [ "${EXIT1}" -eq 0 ] && [ "${EXIT2}" -eq 0 ]; then
    notify "POMO TSP50 baseline dual 完成 ✓" "${SUMMARY}"
else
    notify "POMO TSP50 baseline dual 部分失败 ✗" "${SUMMARY}"
fi

exit $(( EXIT1 | EXIT2 ))
