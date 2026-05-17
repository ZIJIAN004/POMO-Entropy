#!/bin/bash
# ============================================================================
# submit_vrptw_zhuoyi.sh — sbatch 单卡 VRPTW100 POMO-Entropy 实验
#
# 用法（在 zhuoyi 登录节点）：
#   cd /homes/zhuoyi/zijianliu/POMO-Entropy
#   sbatch submit_vrptw_zhuoyi.sh
#
# 配置：
#   qos=long, 1 GPU, softmax-norm + entropy reweight, 200 epoch
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=1
#SBATCH --job-name=pomo_vrptw
#SBATCH --output=/homes/zhuoyi/zijianliu/POMO-Entropy/logs/vrptw_%j.log

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

bash run_vrptw_zhuoyi.sh
