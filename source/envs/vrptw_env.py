"""
VRPTW Environment for POMO

多车辆、时间窗约束、无容量约束。
每辆车从 depot 出发（time=0），到达客户后等待至窗口开启。
返回 depot 后开始新路线（time 重置为 0）。
"""

from dataclasses import dataclass
import torch
from source.problems.vrptw import get_random_vrptw


@dataclass
class Reset_State:
    depot_xy:  torch.Tensor = None   # (batch, 1, 2)
    node_xy:   torch.Tensor = None   # (batch, problem, 2)
    tw_start:  torch.Tensor = None   # (batch, problem+1)
    tw_end:    torch.Tensor = None   # (batch, problem+1)


@dataclass
class Step_State:
    BATCH_IDX:      torch.Tensor = None
    POMO_IDX:       torch.Tensor = None
    selected_count: int          = None
    current_node:   torch.Tensor = None
    current_time:   torch.Tensor = None   # (batch, pomo)
    ninf_mask:      torch.Tensor = None
    finished:       torch.Tensor = None
    n_feasible:     torch.Tensor = None


class VRPTWEnv:
    def __init__(self, problem_size, pomo_size):
        self.problem_size = problem_size
        self.pomo_size    = pomo_size
        self.device       = None

    def load_problems(self, batch_size, device):
        self.batch_size = batch_size
        self.device     = device

        depot_xy, node_xy, tw_start, tw_end = get_random_vrptw(batch_size, self.problem_size)
        self.depot_node_xy = torch.cat((depot_xy, node_xy), dim=1).to(device)   # (batch, N+1, 2)
        self.tw_start      = tw_start.to(device)                                 # (batch, N+1)
        self.tw_end        = tw_end.to(device)                                   # (batch, N+1)

        self.BATCH_IDX = torch.arange(batch_size, device=device)[:, None].expand(batch_size, self.pomo_size)
        self.POMO_IDX  = torch.arange(self.pomo_size, device=device)[None, :].expand(batch_size, self.pomo_size)

        self.reset_state = Reset_State(
            depot_xy=depot_xy.to(device), node_xy=node_xy.to(device),
            tw_start=self.tw_start, tw_end=self.tw_end)
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)

    def reset(self):
        self.selected_count     = 0
        self.current_node       = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0),
                                               dtype=torch.long, device=self.device)

        self.at_the_depot      = torch.ones(self.batch_size, self.pomo_size,
                                             dtype=torch.bool, device=self.device)
        self.current_time      = torch.zeros(self.batch_size, self.pomo_size, device=self.device)
        self.visited_ninf_flag = torch.zeros(self.batch_size, self.pomo_size,
                                              self.problem_size + 1, device=self.device)
        self.ninf_mask         = torch.zeros(self.batch_size, self.pomo_size,
                                              self.problem_size + 1, device=self.device)
        self.finished          = torch.zeros(self.batch_size, self.pomo_size,
                                              dtype=torch.bool, device=self.device)
        return self.reset_state, None, False

    def pre_step(self):
        self._sync()
        return self.step_state, None, False

    def step(self, selected):
        self.selected_count += 1
        self.current_node    = selected
        self.selected_node_list = torch.cat(
            (self.selected_node_list, selected[:, :, None]), dim=2)

        self.at_the_depot = (selected == 0)

        # 计算到达时间
        if self.selected_count >= 2:
            prev_node = self.selected_node_list[:, :, -2]
            prev_xy   = self.depot_node_xy[self.BATCH_IDX, prev_node]     # (batch, pomo, 2)
            curr_xy   = self.depot_node_xy[self.BATCH_IDX, selected]      # (batch, pomo, 2)
            travel    = ((curr_xy - prev_xy) ** 2).sum(dim=2).sqrt()      # (batch, pomo)

            arrival = self.current_time + travel
            # 等待至窗口开启
            tw_s = self.tw_start[self.BATCH_IDX, selected]                # (batch, pomo)
            self.current_time = torch.max(arrival, tw_s)

        # 返回 depot 时 time 重置为 0（新路线）
        self.current_time[self.at_the_depot] = 0.0

        # 更新 visited
        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0   # depot 可重复访问

        # 构建 mask：已访问 + 时间窗不可达
        self.ninf_mask = self.visited_ninf_flag.clone()
        self._mask_time_window()

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished  = self.finished + newly_finished
        self.ninf_mask[:, :, 0][self.finished] = 0

        self._sync()

        done = self.finished.all()
        reward = -self._get_travel_distance() if done else None
        return self.step_state, reward, done

    def _mask_time_window(self):
        """mask 从当前位置出发无法在时间窗内到达的客户节点。"""
        if self.current_node is None:
            return
        curr_xy = self.depot_node_xy[self.BATCH_IDX, self.current_node]   # (batch, pomo, 2)
        all_xy  = self.depot_node_xy[:, None, :, :].expand(
            -1, self.pomo_size, -1, -1)                                    # (batch, pomo, N+1, 2)
        dist_to_all = ((all_xy - curr_xy[:, :, None, :]) ** 2).sum(3).sqrt()  # (batch, pomo, N+1)

        arrival_time = self.current_time[:, :, None] + dist_to_all        # (batch, pomo, N+1)
        too_late = arrival_time > self.tw_end[:, None, :]                  # (batch, pomo, N+1)
        self.ninf_mask[too_late] = float('-inf')

    def _sync(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node   = self.current_node
        self.step_state.current_time   = self.current_time
        self.step_state.ninf_mask      = self.ninf_mask
        self.step_state.finished       = self.finished
        self.step_state.n_feasible     = (self.ninf_mask == 0).sum(dim=2)

    def _get_travel_distance(self):
        idx = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered = all_xy.gather(dim=2, index=idx)
        rolled  = ordered.roll(dims=2, shifts=-1)
        return ((ordered - rolled) ** 2).sum(3).sqrt().sum(2)
