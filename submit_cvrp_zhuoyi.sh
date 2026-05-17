#!/bin/bash
# ============================================================================
# submit_cvrp_zhuoyi.sh — sbatch 单卡 CVRP100 POMO-Entropy monoseg 实验
#
# 用法（在 zhuoyi 登录节点）：
#   cd /homes/zhuoyi/zijianliu/POMO-Entropy
#   sbatch submit_cvrp_zhuoyi.sh
#
# 配置：
#   qos=long, 1 GPU, monoseg baseline (no bucketing),
#   200 epoch × 100k episodes, warmup 10 epochs
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=1
#SBATCH --job-name=pomo_cvrp_mseg
#SBATCH --output=/homes/zhuoyi/zijianliu/POMO-Entropy/logs/cvrp_mseg_%j.log

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

bash run_cvrp_zhuoyi.sh
