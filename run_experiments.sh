#!/bin/bash
# ============================================================================
# run_experiments.sh — POMO-Entropy γ=1.0 reweight 实验启动脚本
#
# 用法（在 yangzhihan 主机上）：
#   cd /Data04/yangzhihan/lzj/POMO-Entropy
#   nohup bash run_experiments.sh > runner.log 2>&1 &
#   tail -f runner.log
#
# 行为：
#   - 检测显存 >10 GiB 的可用 GPU
#   - 取前两张分别跑 TSP100 / CVRP100
#   - 完成 / 异常时通过 Server 酱推送微信
# ============================================================================

set -euo pipefail

PROJECT_DIR="/Data04/yangzhihan/lzj/POMO-Entropy"
PYTHON_BIN="/Data04/yangzhihan/envs/lzj_env/bin/python"
ENV_LIB="/Data04/yangzhihan/envs/lzj_env/lib"
MIN_FREE_MB=10240            # 10 GiB
NEED_GPUS=2                  # TSP + CVRP 各一张
SCKEY="SCT340324Tlw20G3PAJQdqPPHtFAc2J7Qp"

export LD_LIBRARY_PATH="${ENV_LIB}:${LD_LIBRARY_PATH:-}"

LOG_DIR="${PROJECT_DIR}/logs"
mkdir -p "${LOG_DIR}"
TS=$(date +%Y%m%d_%H%M%S)
TSP_LOG="${LOG_DIR}/tsp_g1.0_${TS}.log"
CVRP_LOG="${LOG_DIR}/cvrp_g1.0_${TS}.log"

notify() {
    local title="$1"
    local desp="$2"
    curl -s -m 10 "https://sctapi.ftqq.com/${SCKEY}.send" \
        --data-urlencode "title=${title}" \
        --data-urlencode "desp=${desp}" > /dev/null 2>&1 || true
}

trap 'notify "POMO-Entropy 脚本异常退出" "exit=$? at $(date)\n参考 runner.log"' ERR
trap 'notify "POMO-Entropy 脚本被中断" "SIGINT/SIGTERM at $(date)"' INT TERM

# ── 1) 检测可用 GPU ─────────────────────────────────────────────────────────
echo "[$(date '+%F %T')] 检测可用 GPU (>= ${MIN_FREE_MB} MiB)"

if ! command -v nvidia-smi > /dev/null 2>&1; then
    notify "POMO-Entropy 启动失败" "nvidia-smi 不存在"
    echo "错误: nvidia-smi 不可用"; exit 1
fi

mapfile -t GPU_LINES < <(nvidia-smi --query-gpu=index,memory.free \
                                    --format=csv,noheader,nounits)

available=()
for line in "${GPU_LINES[@]}"; do
    idx=$(echo "${line}" | awk -F',' '{gsub(/ /,""); print $1}')
    free=$(echo "${line}" | awk -F',' '{gsub(/ /,""); print $2}')
    printf "  GPU %s: %s MiB free\n" "${idx}" "${free}"
    if [ "${free}" -ge "${MIN_FREE_MB}" ]; then
        available+=("${idx}")
    fi
done

if [ "${#available[@]}" -lt "${NEED_GPUS}" ]; then
    msg="可用 GPU 不足: 需 ${NEED_GPUS} 张 ≥${MIN_FREE_MB}MiB, 实际仅 ${#available[@]} 张"
    echo "${msg}"
    notify "POMO-Entropy 启动失败" "${msg}"
    exit 1
fi

GPU_TSP="${available[0]}"
GPU_CVRP="${available[1]}"
echo "[$(date '+%F %T')] 选定: TSP -> GPU ${GPU_TSP}, CVRP -> GPU ${GPU_CVRP}"

# ── 2) 启动两个训练进程（后台） ───────────────────────────────────────────────
cd "${PROJECT_DIR}"

echo "[$(date '+%F %T')] 启动 TSP100 (GPU ${GPU_TSP}) -> ${TSP_LOG}"
CUDA_VISIBLE_DEVICES="${GPU_TSP}" nohup "${PYTHON_BIN}" -u Train.py --problem tsp \
    > "${TSP_LOG}" 2>&1 &
TSP_PID=$!

echo "[$(date '+%F %T')] 启动 CVRP100 (GPU ${GPU_CVRP}) -> ${CVRP_LOG}"
CUDA_VISIBLE_DEVICES="${GPU_CVRP}" nohup "${PYTHON_BIN}" -u Train.py --problem cvrp \
    > "${CVRP_LOG}" 2>&1 &
CVRP_PID=$!

echo "[$(date '+%F %T')] TSP_PID=${TSP_PID}  CVRP_PID=${CVRP_PID}"

notify "POMO-Entropy 实验启动" \
"γ=1.0 reweight 实验
TSP100  PID=${TSP_PID}  GPU=${GPU_TSP}
CVRP100 PID=${CVRP_PID} GPU=${GPU_CVRP}
日志: ${LOG_DIR}/"

# ── 3) 等待两个进程结束 ─────────────────────────────────────────────────────
TSP_STATUS=0
CVRP_STATUS=0
wait "${TSP_PID}"  || TSP_STATUS=$?
echo "[$(date '+%F %T')] TSP100 finished (exit=${TSP_STATUS})"
wait "${CVRP_PID}" || CVRP_STATUS=$?
echo "[$(date '+%F %T')] CVRP100 finished (exit=${CVRP_STATUS})"

# ── 4) 完成通知 ─────────────────────────────────────────────────────────────
SUMMARY="TSP100  exit=${TSP_STATUS}  log=${TSP_LOG}
CVRP100 exit=${CVRP_STATUS} log=${CVRP_LOG}"

if [ "${TSP_STATUS}" -eq 0 ] && [ "${CVRP_STATUS}" -eq 0 ]; then
    notify "POMO-Entropy γ=1.0 实验完成 ✓" "${SUMMARY}"
else
    notify "POMO-Entropy γ=1.0 实验异常 ✗" "${SUMMARY}"
fi
