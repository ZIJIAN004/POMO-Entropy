#!/bin/bash
# ============================================================================
# run_cvrp_pb_zhuoyi.sh — CVRP100 monoseg + postbucket 训练 + Server 酱通知
#
# 调用方：submit_cvrp_pb_zhuoyi.sh (sbatch)
# 假设：cwd = /homes/zhuoyi/zijianliu/POMO-Entropy, conda env 已激活
#
# 实验内容：
#   monotonic-segment baseline + bucket post-normalize
#   ΔH_t = (H_t − H[anchor]) − bucket_mean(ΔH_local | n_feasible,at_depot,load)
#   200 epoch, 100k episodes/epoch, warmup 10
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

trap 'notify "POMO-Entropy CVRP monoseg+PB 异常退出" "job=${JOB} exit=$? at $(date)"' ERR
trap 'notify "POMO-Entropy CVRP monoseg+PB 被中断"   "job=${JOB} SIGINT/SIGTERM at $(date)"' INT TERM

echo "[$(date '+%F %T')] job=${JOB}  python=$(which python)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

notify "POMO-Entropy CVRP monoseg+PB 启动" \
"job=${JOB}
problem=cvrp  size=100
monoseg=on  postbucket=on  bucket=3-dim
epoch=200  episodes=100k  warmup=10
log: logs/cvrp_pb_${JOB}.log"

# ── 训练：cvrp100, monoseg + postbucket, 200 epoch ──────────────────────────
python -u Train.py \
    --problem cvrp \
    --size 100 \
    --softmax on \
    --monoseg on \
    --postbucket on \
    --epoch 200

EXIT=$?

if [ "${EXIT}" -eq 0 ]; then
    notify "POMO-Entropy CVRP monoseg+PB 完成 ✓" "job=${JOB} 正常结束 at $(date)"
else
    notify "POMO-Entropy CVRP monoseg+PB 失败 ✗" "job=${JOB} exit=${EXIT} at $(date)"
fi

exit "${EXIT}"
