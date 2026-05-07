"""TSP Environment for POMO"""

from dataclasses import dataclass
import torch
from source.problems.tsp import get_random_tsp


@dataclass
class Reset_State:
    node_xy: torch.Tensor = None   # (batch, problem, 2)


@dataclass
class Step_State:
    BATCH_IDX:      torch.Tensor = None
    POMO_IDX:       torch.Tensor = None
    selected_count: int          = None
    current_node:   torch.Tensor = None
    ninf_mask:      torch.Tensor = None
    n_feasible:     torch.Tensor = None


class TSPEnv:
    def __init__(self, problem_size, pomo_size):
        self.problem_size = problem_size
        self.pomo_size    = pomo_size
        self.device       = None

    def load_problems(self, batch_size, device):
        self.batch_size = batch_size
        self.device     = device

        self.node_xy = get_random_tsp(batch_size, self.problem_size).to(device)

        self.BATCH_IDX = torch.arange(batch_size, device=device)[:, None].expand(batch_size, self.pomo_size)
        self.POMO_IDX  = torch.arange(self.pomo_size, device=device)[None, :].expand(batch_size, self.pomo_size)

        self.reset_state = Reset_State(node_xy=self.node_xy)
        self.step_state  = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)

    def reset(self):
        self.selected_count     = 0
        self.current_node       = None
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0),
                                               dtype=torch.long, device=self.device)
        self.ninf_mask = torch.zeros(self.batch_size, self.pomo_size, self.problem_size,
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

        self.ninf_mask[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')

        self._sync()

        done = (self.selected_count == self.problem_size)
        reward = -self._get_travel_distance() if done else None
        return self.step_state, reward, done

    def _sync(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.current_node   = self.current_node
        self.step_state.ninf_mask      = self.ninf_mask
        self.step_state.n_feasible     = (self.ninf_mask == 0).sum(dim=2)

    def _get_travel_distance(self):
        idx = self.selected_node_list[:, :, :, None].expand(-1, -1, -1, 2)
        seq = self.node_xy[:, None, :, :].expand(-1, self.pomo_size, -1, -1)
        ordered = seq.gather(dim=2, index=idx)
        rolled  = ordered.roll(dims=2, shifts=-1)
        return ((ordered - rolled) ** 2).sum(3).sqrt().sum(2)
