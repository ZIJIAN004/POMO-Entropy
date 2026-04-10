"""
TSP 数据生成（与 UniCOP-Reason/problems/tsp.py 一致）

坐标：uniform [0,1]^2，共 problem_size 个节点（无 depot 区分，POMO 对称）
"""

import torch


def get_random_tsp(batch_size, problem_size):
    """返回 (batch, problem_size, 2) 的节点坐标。"""
    return torch.rand(batch_size, problem_size, 2)
