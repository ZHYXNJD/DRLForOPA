import functools
import sys
from collections import defaultdict

import gymnasium as gym
import pandas as pd
from gymnasium import spaces
from gymnasium.spaces import Tuple, Dict, Discrete, MultiBinary, MultiDiscrete
import numpy as np
import static_data as sd
from copy import deepcopy

# 先做个占位 后面再导包
total_request = 2120
window_time = 195
# window_time = 1440

req_info = pd.read_csv(sd.req_info_path)
req_revenue = list(np.array((req_info['parking_t'].fillna(0) + req_info['char_t'].fillna(0)).values, dtype=int))
req_type = list(np.array(req_info['charge_label'], dtype=int))
rmk = np.array(pd.read_csv(sd.r_mk_path))  # 请求时间矩阵

park_lot_num = sd.total_slot
park_slot_index = sd.ops_index
charge_slot_index = sd.cps_index
slot_index = [park_slot_index, charge_slot_index]  # 两类索引

state_list = []
action_list = []
reward_list = []
next_state_list = []
cumulative_reward_list = []


class ActionSpace:
    def __init__(self, choice):
        self.assign = MultiDiscrete([park_lot_num,])
        self.choice = choice

    def sample(self):
        return np.random.choice(self.choice)


class ObservationSpace:
    def __init__(self):
        # demand 的维度应该是多少？？？ 是所有数量的request还是单个的
        self.observation_space = Dict({"demand": spaces.Box(low=0, high=1, shape=(1, window_time), dtype=np.short),
                                       "supply": spaces.Box(low=0, high=1, shape=(park_lot_num, window_time),
                                                            dtype=np.short)})


class OPA(gym.Env):

    def __init__(self, render_mode=None):
        self.infos = None
        self.termination = False
        self.total_request = total_request
        self.rewards = 0
        self._cumulative_rewards = 0
        self.dones = False
        self.request_num = 0
        self.states = None
        self.observations = None
        self.action = None

    def reset(self, **kwargs):
        """
        -rewards
        -_cumulative_rewards
        -termination
        -truncations
        -infos
        And must set up the environment so that render() step() and observe() can be called without issues
        :param **kwargs:
        """
        super().reset(seed=None)
        self.rewards = 0
        self._cumulative_rewards = 0
        self.termination = False
        self.states = {"demand": rmk[0], "type":req_type[0],"supply": np.zeros((park_lot_num, window_time))}
        self.request_num = 0
        self.infos = {}
        self.request_num += 1
        return self.states

    def step(self, action):
        """
        step(action) takes in an action for the agent and needs to update
        - rewards
        -_cumulative_rewards
        -terminations
        -truncations
        -infos
        And any internal state used by observe() or render()
        """
        if self.request_num < self.total_request:
            if action == 324:
                # 如果决绝了请求 不给予奖励
                self.rewards = 0
                print(f"已拒绝该请求:{self.request_num-1}")

            else:
                self.rewards = req_revenue[self.request_num-1]

                # 更新states
                self.states["supply"][action] = self.states["supply"][action] + self.states["demand"]

            next_states = {"demand": rmk[self.request_num], "type":req_type[self.request_num], "supply": self.states["supply"]}
            self.states = next_states
            self._cumulative_rewards += self.rewards
            self.infos = {"req_id": self.request_num-1, "rewards": self.rewards,"_cumulative_rewards": self._cumulative_rewards}
            print(self.infos)
            self.request_num += 1
            return next_states, self.rewards, self._cumulative_rewards, self.termination
        else:
            self.termination = True
            print("所有请求分配完毕，本次episode结束...")
            return self.states, self.rewards, self._cumulative_rewards, self.termination


# 更新动作空间
def get_mask(agent):
    demand = agent.states["demand"]
    supply = agent.states["supply"]
    demand_type = req_type[agent.request_num]
    temp_supply = supply[slot_index[demand_type]] + demand
    row_index = np.where(np.all(temp_supply < 2, axis=1))[0]  # 返回可用的泊位 作为新的动作空间
    mask_array = np.array([1 if i in row_index else 0 for i in range(park_lot_num)], dtype=np.int8)
    mask_tuple = (mask_array,)
    return mask_tuple


def get_choice_set_base_state(state):
    demand = state["demand"]
    supply = state["supply"]
    demand_type = state["type"]
    temp_data = pd.DataFrame(supply).loc[slot_index[demand_type]] + demand
    choice_list = temp_data[temp_data.apply(lambda row:all(x < 2 for x in row), axis=1)].index.values   # 返回可用的泊位 作为新的动作空间
    # 字典是不可哈希的 因此不能作为键来使用 转化为可哈希的对象
    state_as_key = {"demand":tuple(demand),"type":demand_type,"supply":tuple(supply)}
    return choice_list,state_as_key


