"""CVRP Environment for POMO"""

from dataclasses import dataclass
import torch
from source.problems.cvrp import get_random_cvrp


@dataclass
class Reset_State:
    depot_xy:    torch.Tensor = None   # (batch, 1, 2)
    node_xy:     torch.Tensor = None   # (batch, problem, 2)
    node_demand: torch.Tensor = None   # (batch, problem)


@dataclass
class Step_State:
    BATCH_IDX:              torch.Tensor = None
    POMO_IDX:               torch.Tensor = None
    selected_count:         int          = None
    load:                   torch.Tensor = None
    current_node:           torch.Tensor = None
    ninf_mask:              torch.Tensor = None
    finished:               torch.Tensor = None
    n_feasible:             torch.Tensor = None
    visited_customer_count: torch.Tensor = None   # number of non-depot visits so far


class CVRPEnv:
    def __init__(self, problem_size, pomo_size):
        self.problem_size = problem_size
        self.pomo_size    = pomo_size
        self.device       = None

    def load_problems(self, batch_size, device):
        self.batch_size = batch_size
        self.device     = device

        depot_xy, node_xy, node_demand = get_random_cvrp(batch_size, self.problem_size)
        depot_xy    = depot_xy.to(device)
        node_xy     = node_xy.to(device)
        node_demand = node_demand.to(device)

        self.depot_node_xy     = torch.cat((depot_xy, node_xy), dim=1)
        depot_demand           = torch.zeros(batch_size, 1, device=device)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)

        self.BATCH_IDX = torch.arange(batch_size, device=device)[:, None].expand(batch_size, self.pomo_size)
        self.POMO_IDX  = torch.arange(self.pomo_size, device=device)[None, :].expand(batch_size, self.pomo_size)

        self.reset_state = Reset_State(depot_xy=depot_xy, node_xy=node_xy, node_demand=node_demand)
        self.step_state  = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)

    def reset(self):
        self.selected_count     = 0
        self.current_node       = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0),
                                               dtype=torch.long, device=self.device)

        self.at_the_depot      = torch.ones(self.batch_size, self.pomo_size,
                                             dtype=torch.bool, device=self.device)
        self.load              = torch.ones(self.batch_size, self.pomo_size, device=self.device)
        self.visited_ninf_flag = torch.zeros(self.batch_size, self.pomo_size,
                                              self.problem_size + 1, device=self.device)
        self.ninf_mask         = torch.zeros(self.batch_size, self.pomo_size,
                                              self.problem_size + 1, device=self.device)
        self.finished          = torch.zeros(self.batch_size, self.pomo_size,
                                              dtype=torch.bool, device=self.device)
        self.visited_customer_count = torch.zeros(self.batch_size, self.pomo_size,
                                                   device=self.device)
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

        # 累计客户访问数（depot 不计）
        self.visited_customer_count = self.visited_customer_count + (~self.at_the_depot).float()

        demand_list     = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        selected_demand = demand_list.gather(dim=2, index=selected[:, :, None]).squeeze(2)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0

        self.ninf_mask = self.visited_ninf_flag.clone()
        demand_too_large = self.load[:, :, None] + 1e-5 < demand_list
        self.ninf_mask[demand_too_large] = float('-inf')

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        self.finished  = self.finished + newly_finished
        self.ninf_mask[:, :, 0][self.finished] = 0

        self._sync()

        done = self.finished.all()
        reward = -self._get_travel_distance() if done else None
        return self.step_state, reward, done

    def _sync(self):
        self.step_state.selected_count         = self.selected_count
        self.step_state.load                   = self.load
        self.step_state.current_node           = self.current_node
        self.step_state.ninf_mask              = self.ninf_mask
        self.step_state.finished               = self.finished
        self.step_state.n_feasible             = (self.ninf_mask == 0).sum(dim=2)
        self.step_state.visited_customer_count = self.visited_customer_count

    def _get_travel_distance(self):
        idx = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        all_xy = self.depot_node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered = all_xy.gather(dim=2, index=idx)
        rolled  = ordered.roll(dims=2, shifts=-1)
        return ((ordered - rolled) ** 2).sum(3).sqrt().sum(2)
