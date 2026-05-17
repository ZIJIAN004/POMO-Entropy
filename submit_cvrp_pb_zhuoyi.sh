#!/bin/bash
# ============================================================================
# submit_cvrp_pb_zhuoyi.sh — sbatch CVRP100 monoseg + postbucket
#
# 用法（在 zhuoyi 登录节点）：
#   cd /homes/zhuoyi/zijianliu/POMO-Entropy
#   sbatch submit_cvrp_pb_zhuoyi.sh
#
# 配置：
#   qos=long, 1 GPU
#   monoseg (trajectory trend) + postbucket (subtract bucket-mean of ΔH_local)
#   bucket 3-dim: (n_feasible, at_depot, load_bin) — no vis_ratio
#   200 epoch × 100k episodes, warmup 10
# ============================================================================
#SBATCH --qos long
#SBATCH --gpus=1
#SBATCH --job-name=pomo_cvrp_pb
#SBATCH --output=/homes/zhuoyi/zijianliu/POMO-Entropy/logs/cvrp_pb_%j.log

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

bash run_cvrp_pb_zhuoyi.sh
