import gc
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from dqn_agent import Agent
from OPA import OPA
import static_data as sd

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

total_slot = sd.total_slot  # 泊位总数量
park_slot_index = sd.ops_index  # 普通泊位索引
charge_slot_index = sd.cps_index  # 充电桩索引
slot_index = [park_slot_index, charge_slot_index]  # 两类索引集合
window_time = sd.window_time  # 将时间离散化后的时间间隔总数 这里是195（15min为单位）
total_request = 2120  # 2000个普通请求+120个充电请求

req_info = pd.read_csv(sd.req_info_path)
req_revenue = np.array((req_info['parking_t'].fillna(0) + req_info['char_t'].fillna(0)).values, dtype=int)
req_type = np.array(req_info["charge_label"], dtype=int)
rmk = np.array(pd.read_csv(sd.r_mk_path))

# 即时决策的话
# 停车场泊位供应状态 + 一个需求信息 + 需求种类
state_size = (total_slot + 1) * window_time + 1
# 动作空间为停车泊位的索引
# 需要加1 代表不分配任何泊位
action_size = total_slot + 1

agent = Agent(state_size=state_size, action_size=action_size, seed=0)
env = OPA()


def new_state(env_state):
    return np.concatenate([env_state["supply"].flatten(), env_state["demand"].flatten(),env_state["type"].flatten()])


def get_invalid_actions(env_state):
    demand = env_state["demand"]
    supply = env_state["supply"]
    demand_type = env_state["type"]
    temp_data = pd.DataFrame(supply).loc[slot_index[demand_type]] + demand
    valid_choice = temp_data[temp_data.apply(lambda row: all(x < 2 for x in row), axis=1)].index.values  # 返回可用的泊位 作为新的动作空间
    invalid_choice = list[set(np.arange(0,action_size))-set(valid_choice)]
    return invalid_choice


def dqn(n_episode=15, episode_length=total_request, eps_start=1.0, eps_end=0.01, eps_decay=0.995):
    """

    :param n_episode: max number of training episodes
    :param episode_length:
    :param eps_start:
    :param eps_end:
    :param eps_decay:
    :return:
    """

    scores = []
    eps = eps_start
    for i_episode in range(1, n_episode + 1):
        state = env.reset()
        env_state = deepcopy(state)  # 因为是字典 需要深拷贝 否则会修改原state  这个state仍然是字典 可以通过关键字得到对应的值
        agent_state = new_state(env_state)
        score = 0
        for t in range(episode_length):
            curr_invalid_choice = get_invalid_actions(env_state)
            action = agent.act(agent_state,curr_invalid_choice,eps)
            next_state, reward, cum_rewards, done = env.step(action)
            next_env_state = deepcopy(next_state)
            next_invalid_choice = get_invalid_actions(next_env_state)
            next_agent_state = new_state(next_env_state)
            agent.step(agent_state, action, reward, next_agent_state, done,curr_invalid_choice,next_invalid_choice)
            agent_state = next_agent_state
            env_state = next_env_state
            score += reward
            if done:
                break
        env.close()
        gc.collect()
        scores.append(score)
        eps = max(eps_end, eps_decay * eps)
        if i_episode % 2 == 0:
            print('\rEpisode {}\t Score: {:.2f}'.format(i_episode, score), end="")
        if i_episode % 2 == 0:
            torch.save(agent.qnetwork_local.state_dict(), 'checkpoint.pth')
    return scores


scores = dqn()

# plot the scores
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(len(scores)), scores)
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.show()
