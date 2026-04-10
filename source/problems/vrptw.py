"""
VRPTW 数据生成（与 UniCOP-Reason/problems/vrptw.py 一致）

坐标：uniform [0,1]^2
时间窗：
  depot: [0, 1e9]
  customer i: a_i = dist(depot, i) + uniform(0, 0.2)
              b_i = a_i + uniform(0.3, 0.5)
无需求/容量约束。
"""

import torch


def get_random_vrptw(batch_size, problem_size):
    """
    返回:
        depot_xy:    (batch, 1, 2)
        node_xy:     (batch, problem_size, 2)
        tw_start:    (batch, problem_size+1)   depot=0, customers=a_i
        tw_end:      (batch, problem_size+1)   depot=1e9, customers=b_i
    """
    depot_xy = torch.rand(batch_size, 1, 2)
    node_xy = torch.rand(batch_size, problem_size, 2)

    # dist(depot, customer_i)
    dist_depot = (node_xy - depot_xy).pow(2).sum(dim=2).sqrt()  # (batch, problem_size)

    a = dist_depot + torch.rand(batch_size, problem_size) * 0.2
    width = 0.3 + torch.rand(batch_size, problem_size) * 0.2   # uniform(0.3, 0.5)
    b = a + width

    # 拼接 depot 的时间窗
    depot_tw_start = torch.zeros(batch_size, 1)
    depot_tw_end = torch.full((batch_size, 1), 1e9)

    tw_start = torch.cat([depot_tw_start, a], dim=1)    # (batch, problem_size+1)
    tw_end = torch.cat([depot_tw_end, b], dim=1)         # (batch, problem_size+1)

    return depot_xy, node_xy, tw_start, tw_end
