#!/bin/bash
# ============================================================================
# submit_zhuoyi.sh — generic POMO training submission on zhuoyi
#
# 用法：sbatch 时把任务名/输出路径用命令行 override，Python 参数作为位置参数：
#
#   sbatch -J <name> -o logs/<name>_%j.log submit_zhuoyi.sh <python args...>
#
# 例：
#   sbatch -J tsp_method_s1 -o logs/tsp_method_s1_%j.log \
#          submit_zhuoyi.sh --problem tsp --softmax on --epoch 200 --seed 1
#
# 包含 Server 酱启动/完成/异常通知。
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=1

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
NAME="${SLURM_JOB_NAME:-pomo}"

notify() {
    curl -s -m 10 "https://sctapi.ftqq.com/${SCKEY}.send" \
        --data-urlencode "title=$1" \
        --data-urlencode "desp=$2" > /dev/null 2>&1 || true
}

trap 'notify "POMO ${NAME} 异常退出" "job=${JOB} exit=$? at $(date)"' ERR
trap 'notify "POMO ${NAME} 被中断"   "job=${JOB} SIGINT/SIGTERM at $(date)"' INT TERM

echo "[$(date '+%F %T')] job=${JOB} name=${NAME}  python=$(which python)"
echo "args: $*"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

notify "POMO ${NAME} 启动" \
"job=${JOB}
args: $*
log: logs/${NAME}_${JOB}.log"

python -u Train.py "$@"
EXIT=$?

if [ "${EXIT}" -eq 0 ]; then
    notify "POMO ${NAME} 完成 ✓" "job=${JOB} 正常结束 at $(date)"
else
    notify "POMO ${NAME} 失败 ✗" "job=${JOB} exit=${EXIT} at $(date)"
fi

exit "${EXIT}"
