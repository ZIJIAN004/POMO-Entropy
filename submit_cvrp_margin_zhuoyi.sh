#!/bin/bash
# ============================================================================
# submit_cvrp_margin_zhuoyi.sh — CVRP100 prob-margin + 4-dim bucket, 200 ep
#
# 信号源换：entropy → prob margin = p(top-1) − p(top-2) ∈ [0,1] (scale-free)
# 其余保持：4 维 bucket (n_f, at_depot, load, vis_ratio), softmax γ=0.3.
# log 里还会出现 OLS R² 诊断：F2=主效应, F5=+交叉/二次/三次项。
#
# 用法（zhuoyi 登录节点）：
#   cd /homes/zhuoyi/zijianliu/POMO-Entropy
#   sbatch submit_cvrp_margin_zhuoyi.sh
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=1
#SBATCH --job-name=cvrp_margin
#SBATCH --output=/homes/zhuoyi/zijianliu/POMO-Entropy/logs/cvrp_margin_%j.log

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

trap 'notify "POMO CVRP100 margin 异常" "job=${JOB} exit=$? at $(date)"' ERR
trap 'notify "POMO CVRP100 margin 中断" "job=${JOB} at $(date)"' INT TERM

echo "[$(date '+%F %T')] job=${JOB}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

notify "POMO CVRP100 margin 启动" \
"job=${JOB}
solo (1 GPU), seed=1
problem=cvrp size=100 softmax=on signal=margin gamma=0.3 epoch=200
信号源换：prob margin (top1-top2 prob)
新加 OLS R² 诊断 (F2/F5)"

python -u Train.py --problem cvrp --size 100 --softmax on --signal margin --gamma 0.3 --epoch 200 --seed 1

EXIT=$?

if [ "${EXIT}" -eq 0 ]; then
    notify "POMO CVRP100 margin 完成 ✓" "job=${JOB} 正常结束 at $(date)"
else
    notify "POMO CVRP100 margin 失败 ✗" "job=${JOB} exit=${EXIT} at $(date)"
fi

exit "${EXIT}"
