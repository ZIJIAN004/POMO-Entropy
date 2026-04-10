"""
CVRP 数据生成（与 UniCOP-Reason/problems/cvrp.py 一致）

坐标：uniform [0,1]^2
需求：integers(1,10) / demand_scaler(n)，depot 需求为 0
容量：1.0
"""

import torch


def _demand_scaler(n: int) -> int:
    """按问题规模返回需求归一化系数，使期望路线数约为 3~6。"""
    table = {10: 15, 20: 30, 50: 40, 100: 50}
    return table.get(n, max(10, round(n * 5 / 3)))


def get_random_cvrp(batch_size, problem_size):
    """
    返回:
        depot_xy:    (batch, 1, 2)
        node_xy:     (batch, problem_size, 2)
        node_demand: (batch, problem_size)   归一化需求，值域 (0, 1)
    """
    depot_xy = torch.rand(batch_size, 1, 2)
    node_xy = torch.rand(batch_size, problem_size, 2)
    scaler = _demand_scaler(problem_size)
    node_demand = torch.randint(1, 10, size=(batch_size, problem_size)) / float(scaler)
    return depot_xy, node_xy, node_demand
