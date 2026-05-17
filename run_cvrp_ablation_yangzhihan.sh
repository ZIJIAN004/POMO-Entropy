#!/bin/bash
# ============================================================================
# run_cvrp_ablation_yangzhihan.sh — CVRP100 ablation: drop vis_ratio + low-grp mask
#
# Ablation vs. baseline 4-dim softmax run (the 14-ep CVRP log analyzed earlier):
#   --vis-ratio off        : 4-dim bucket → 3-dim (n_feasible, at_depot, load_bin).
#                            top3=0.022 under 4-dim suggested over-fine binning.
#   --lowg-thresh 0.05     : strip rw eligibility from buckets with grp_mean<0.05.
#                            lowG(<.05)=0.08 under 4-dim → ~8% of rw steps were
#                            in near-deterministic buckets carrying noise.
#
# 用法（在 yangzhihan 上）：
#   cd /Data04/yangzhihan/lzj/POMO-Entropy
#   nohup bash run_cvrp_ablation_yangzhihan.sh <GPU_ID> > runner_ablation.log 2>&1 &
#   tail -f runner_ablation.log
#
#   GPU_ID 可省略，默认选第一张 free>=10GiB 的 GPU。
# ============================================================================

set -euo pipefail

PROJECT_DIR="/Data04/yangzhihan/lzj/POMO-Entropy"
PYTHON_BIN="/Data04/yangzhihan/envs/lzj_env/bin/python"
ENV_LIB="/Data04/yangzhihan/envs/lzj_env/lib"
MIN_FREE_MB=10240
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

export LD_LIBRARY_PATH="${ENV_LIB}:${LD_LIBRARY_PATH:-}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
CVRP_LOG="${LOG_DIR}/cvrp_ablation_noVR_lowg05_${TS}.log"

notify() {
    local title="$1"
    local desp="$2"
    curl -s -m 10 "https://sctapi.ftqq.com/${SCKEY}.send" \
        --data-urlencode "title=${title}" \
        --data-urlencode "desp=${desp}" > /dev/null 2>&1 || true
}

trap 'notify "POMO-Entropy CVRP ablation 异常退出" "exit=$? at $(date)"' ERR
trap 'notify "POMO-Entropy CVRP ablation 被中断"   "SIGINT/SIGTERM at $(date)"' INT TERM

# ── GPU 选择 ─────────────────────────────────────────────────────────────────
if [ "$#" -ge 1 ]; then
    GPU_ID="$1"
    echo "[$(date '+%F %T')] 使用指定 GPU: ${GPU_ID}"
else
    mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,memory.free \
                                        --format=csv,noheader,nounits)
    GPU_ID=""
    for line in "${GPU_LINES[@]}"; do
        idx=$(echo "${line}" | awk -F',' '{gsub(/ /,""); print $1}')
        free=$(echo "${line}" | awk -F',' '{gsub(/ /,""); print $2}')
        if [ "${free}" -ge "${MIN_FREE_MB}" ]; then
            GPU_ID="${idx}"; break
        fi
    done
    if [ -z "${GPU_ID}" ]; then
        notify "POMO-Entropy CVRP ablation 启动失败" "无可用 GPU (≥${MIN_FREE_MB}MiB)"
        exit 1
    fi
    echo "[$(date '+%F %T')] 自动选 GPU: ${GPU_ID}"
fi

cd "${PROJECT_DIR}"

notify "POMO-Entropy CVRP ablation 启动" \
"problem=cvrp size=100  softmax=on
ablation: --vis-ratio off  --lowg-thresh 0.05
GPU=${GPU_ID}  epoch=200
log: ${CVRP_LOG}"

# ── 训练：CVRP100, softmax + no vis_ratio + lowg mask 0.05, 200 epoch ────────
CUDA_VISIBLE_DEVICES="${GPU_ID}" "${PYTHON_BIN}" -u Train.py \
    --problem cvrp \
    --softmax on \
    --vis-ratio off \
    --lowg-thresh 0.05 \
    --epoch 200 \
    > "${CVRP_LOG}" 2>&1
EXIT=$?

if [ "${EXIT}" -eq 0 ]; then
    notify "POMO-Entropy CVRP ablation 完成 ✓" "GPU=${GPU_ID} 正常结束 at $(date)"
else
    notify "POMO-Entropy CVRP ablation 失败 ✗" "GPU=${GPU_ID} exit=${EXIT} at $(date)"
fi

exit "${EXIT}"
