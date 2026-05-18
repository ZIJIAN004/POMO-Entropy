#!/bin/bash
# ============================================================================
# submit_tsp50_sm_solo_zhuoyi.sh — TSP50 softmax method, 1 GPU, 1 process, 100 ep
#
# 单卡独占跑一个 seed（默认 seed 3），不与其它进程争 GPU，作为 dual-run 的
# 对照（验证同卡共享是否影响性能）。
#
# 用法（zhuoyi 登录节点）：
#   cd /homes/zhuoyi/zijianliu/POMO-Entropy
#   sbatch submit_tsp50_sm_solo_zhuoyi.sh
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=1
#SBATCH --job-name=tsp50_sm_solo
#SBATCH --output=/homes/zhuoyi/zijianliu/POMO-Entropy/logs/tsp50_sm_solo_%j.log

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

trap 'notify "POMO TSP50 softmax solo 异常" "job=${JOB} exit=$? at $(date)"' ERR
trap 'notify "POMO TSP50 softmax solo 中断" "job=${JOB} at $(date)"' INT TERM

echo "[$(date '+%F %T')] job=${JOB}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

notify "POMO TSP50 softmax solo 启动" \
"job=${JOB}
solo (1 GPU, 1 process), seed=3
problem=tsp size=50 softmax=on epoch=100"

python -u Train.py --problem tsp --size 50 --softmax on --epoch 100 --seed 3

EXIT=$?

if [ "${EXIT}" -eq 0 ]; then
    notify "POMO TSP50 softmax solo 完成 ✓" "job=${JOB} 正常结束 at $(date)"
else
    notify "POMO TSP50 softmax solo 失败 ✗" "job=${JOB} exit=${EXIT} at $(date)"
fi

exit "${EXIT}"
