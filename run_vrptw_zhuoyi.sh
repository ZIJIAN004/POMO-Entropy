#!/bin/bash
# ============================================================================
# run_vrptw_zhuoyi.sh — 实际启动 VRPTW100 训练 + Server 酱通知
#
# 调用方：submit_vrptw_zhuoyi.sh (sbatch)
# 假设：cwd = /homes/zhuoyi/zijianliu/POMO-Entropy, conda env 已激活
# ============================================================================

set -euo pipefail

SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"
JOB="${SLURM_JOB_ID:-local}"

notify() {
    local title="$1"
    local desp="$2"
    curl -s -m 10 "https://sctapi.ftqq.com/${SCKEY}.send" \
        --data-urlencode "title=${title}" \
        --data-urlencode "desp=${desp}" > /dev/null 2>&1 || true
}

trap 'notify "POMO-Entropy VRPTW 异常退出" "job=${JOB} exit=$? at $(date)"' ERR
trap 'notify "POMO-Entropy VRPTW 被中断"   "job=${JOB} SIGINT/SIGTERM at $(date)"' INT TERM

echo "[$(date '+%F %T')] job=${JOB}  python=$(which python)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

notify "POMO-Entropy VRPTW 启动" \
"job=${JOB}
problem=vrptw  size=100  softmax=on  epoch=200
log: logs/vrptw_${JOB}.log"

# ── 训练：vrptw100, softmax-norm reweight, 200 epoch ─────────────────────────
python -u Train.py \
    --problem vrptw \
    --size 100 \
    --softmax on \
    --epoch 200

EXIT=$?

if [ "${EXIT}" -eq 0 ]; then
    notify "POMO-Entropy VRPTW 完成 ✓" "job=${JOB} 正常结束 at $(date)"
else
    notify "POMO-Entropy VRPTW 失败 ✗" "job=${JOB} exit=${EXIT} at $(date)"
fi

exit "${EXIT}"
