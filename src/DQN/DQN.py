import gc
import json
from pathlib import Path
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from dqn_agent import Agent,BATCH_SIZE
from OPA import OPA
import static_data as sd
from torch.utils.tensorboard import SummaryWriter
import os

total_slot = sd.total_slot  # 泊位总数量
park_slot_index = sd.ops_index  # 普通泊位索引
charge_slot_index = sd.cps_index  # 充电桩索引
slot_index = [park_slot_index, charge_slot_index]  # 两类索引集合
window_time = sd.window_time  # 将时间离散化后的时间间隔总数 这里是195（15min为单位）
total_request = 2120  # 2000个普通请求+120个充电请求

# req_info = pd.read_csv(sd.req_info_path)
# req_revenue = np.array((req_info['parking_t'].fillna(0) + req_info['char_t'].fillna(0)).values, dtype=int)
# req_type = np.array(req_info["charge_label"], dtype=int)
# rmk = np.array(pd.read_csv(sd.r_mk_path))


# 即时决策的话
# 停车场泊位供应状态 + 一个需求信息 + 需求种类
state_size = (total_slot + 1) * window_time + 1
# 动作空间为停车泊位的索引
# 需要加1 代表不分配任何泊位
action_size = total_slot + 1

agent = Agent(state_size=state_size, action_size=action_size, seed=0)
env = OPA()
writer = SummaryWriter('./newlog')

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


def dqn(n_episode=30, episode_start=1,episode_length=total_request, eps_start=1.0, eps_end=0.01, eps_decay=0.995,load_path=None):
    """

    :param n_episode: max number of training episodes
    :param episode_length:
    :param eps_start:
    :param eps_end:
    :param eps_decay:
    :return:
    """
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / 'new_weights'
    data_dir = script_dir / 'new_data_dir'

    eps = eps_start

    if load_path is not None:
        agent.qnetwork_local.load_state_dict(torch.load(load_path))
        print("已从第{}次开始加载".format(episode_start))

    for i_episode in range(episode_start+1, n_episode + 1):
        state = env.reset()
        env_state = deepcopy(state)  # 因为是字典 需要深拷贝 否则会修改原state  这个state仍然是字典 可以通过关键字得到对应的值
        agent_state = new_state(env_state)
        score = 0
        loss = 0
        for t in range(episode_length):
            curr_invalid_choice = get_invalid_actions(env_state)
            action = agent.act(agent_state,curr_invalid_choice,eps)
            next_state, reward, done,info = env.step(action)
            # 将信息写入txt文件
            with open(data_dir / f'episode_{i_episode}.txt',mode='a',encoding='utf-8') as f:
                f.write(json.dumps(info)+'\n')
            score += reward
            if t < episode_length-1:
                next_env_state = deepcopy(next_state)
                next_invalid_choice = get_invalid_actions(next_env_state)
                next_agent_state = new_state(next_env_state)
                loss += agent.step(agent_state, action, reward, next_agent_state, done,curr_invalid_choice,next_invalid_choice)
                agent_state = next_agent_state
                env_state = next_env_state
                # print("episode:{},step{},loss:{}".format(i_episode,t,loss))
            if done:
                f.close()
                break
        loss = loss / (BATCH_SIZE*episode_length) / 1e10
        print("episode_average_loss:{}".format(loss))
        with open(data_dir / 'scores.txt',mode='a',encoding='utf-8') as f:
            f.write('episode_{}:'+str(score)+'\n'.format(i_episode))
            f.close()
        writer.add_scalar('score:',score,i_episode)
        writer.close()
        writer.add_scalar('loss:',loss,i_episode)
        writer.close()
        eps = max(eps_end, eps_decay * eps)
        print('\rEpisode {}\t Score: {:.2f}\n'.format(i_episode, score), end="")
        gc.collect()
        if i_episode % 5 == 0:
            torch.save(agent.qnetwork_local.state_dict(), log_dir / f'episode_{i_episode}_checkpoint.pth')


def check_log():
    script_dir = Path(__file__).resolve().parent
    log_dir = script_dir / 'new_weights'
    weights = os.listdir(log_dir)
    if len(weights) > 0:
        num_of_episode = int(weights[-1].split("_")[1])
        episode_path = log_dir / weights[-1]
        eps_end = 0.01
        eps_decay = 0.995
        eps_start = max(eps_end,pow(eps_decay,num_of_episode))
        print("episode_path:{}".format(episode_path))
        dqn(n_episode=150,episode_start=num_of_episode,episode_length=total_request,eps_start=eps_start,eps_end=eps_end,eps_decay=eps_decay,load_path=episode_path)
    else:
        dqn()


check_log()

# scores = dqn()


# plot the scores
# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.plot(np.arange(len(scores)), scores)
# plt.ylabel('Score')
# plt.xlabel('Episode #')
# plt.show()

