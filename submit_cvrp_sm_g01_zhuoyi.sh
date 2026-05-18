#!/bin/bash
# ============================================================================
# submit_cvrp_sm_g01_zhuoyi.sh — CVRP100 softmax, γ=0.1 solo, 200 ep
#
# γ sweep 下行：0.3 持平 baseline, 0.7 显著差（noise amplification）→ 试更小
# γ 压制 ΔH heavy-tail outlier 的影响。
#
# 用法（zhuoyi 登录节点）：
#   cd /homes/zhuoyi/zijianliu/POMO-Entropy
#   sbatch submit_cvrp_sm_g01_zhuoyi.sh
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=1
#SBATCH --job-name=cvrp_sm_g01
#SBATCH --output=/homes/zhuoyi/zijianliu/POMO-Entropy/logs/cvrp_sm_g01_%j.log

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

trap 'notify "POMO CVRP100 softmax γ=0.1 异常" "job=${JOB} exit=$? at $(date)"' ERR
trap 'notify "POMO CVRP100 softmax γ=0.1 中断" "job=${JOB} at $(date)"' INT TERM

echo "[$(date '+%F %T')] job=${JOB}"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

notify "POMO CVRP100 softmax γ=0.1 启动" \
"job=${JOB}
solo (1 GPU), seed=1
problem=cvrp size=100 softmax=on gamma=0.1 epoch=200"

python -u Train.py --problem cvrp --size 100 --softmax on --gamma 0.1 --epoch 200 --seed 1

EXIT=$?

if [ "${EXIT}" -eq 0 ]; then
    notify "POMO CVRP100 softmax γ=0.1 完成 ✓" "job=${JOB} 正常结束 at $(date)"
else
    notify "POMO CVRP100 softmax γ=0.1 失败 ✗" "job=${JOB} exit=${EXIT} at $(date)"
fi

exit "${EXIT}"
