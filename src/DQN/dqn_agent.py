import numpy as np
import random
from collections import namedtuple, deque

from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e6)  # 存储的最大数据量
# BATCH_SIZE = 64  # 每次训练的数据量
BATCH_SIZE = 32  # 每次训练的数据量
GAMMA = 0.99  # 衰减率 discount factor
TAU = 1e-3  # target network 软更新参数
LR = 5e-4  # 网络学习率
UPDATE_EVERY = 1  # 更新网络的频率
# UPDATE_EVERY = 4  # 更新网络的频率

# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class ReplayBuffer:
    """
    fixed size buffer to store experience tuples
    """

    def __init__(self, action_size, buffer_size, bath_size, seed):
        """
        Param
        ======
        :param action_size: dimension of each action
        :param buffer_size: maximum size of buffer
        :param bath_size: size of each training batch
        :param seed: random seed
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = bath_size
        self.experiences = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done):
        e = self.experiences(state, action, reward, next_state, done)
        self.memory.append(e)

    def sample(self):
        """
        :return: randomly sample a batch of experiences from memory
        """
        # experiences = random.sample(self.memory, k=self.batch_size)
        # states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        # actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        # rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        # next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
        #     device)
        # dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
        #     device)

        experiences = random.sample(self.memory, k=self.batch_size)
        states = torch.tensor(np.vstack([e.state for e in experiences if e is not None])).float()
        actions = torch.tensor(np.vstack([e.action for e in experiences if e is not None])).long()
        rewards = torch.tensor(np.vstack([e.reward for e in experiences if e is not None])).float()
        next_states = torch.tensor(np.vstack([e.next_state for e in experiences if e is not None])).float()
        dones = torch.tensor(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float()

        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)


class Agent:

    def __init__(self, state_size, action_size, seed):

        """
        Params
        ======
        :param state_size: (ndarray) dimension of each state,it should include parking lot's occupancy and demand request
        :param action_size: (vector) dimension of each action, the probs of choosing each parking slot  
        :param seed: (int) random seed
        :return: 
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = seed

        # Q-Network
        # self.qnetwork_local = QNetwork(state_size, action_size, seed).to(device)
        # self.qnetwork_target = QNetwork(state_size, action_size, seed).to(device)
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ReplayBuffer(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # initialize time step for updating every UPDATE_EVERY steps
        self.t_step = 0

    def step(self, state, action, reward, next_state, done,curr_invalid_choice,next_invalid_choice):

        # save memory in replay buffer
        self.memory.add(state, action, reward, next_state, done)

        # learn every UPDATE_EVERY time step
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # if enough memory are available in memory, get random substate and learn
            if len(self.memory) > BATCH_SIZE:
                experiences = self.memory.sample()
                self.learn(experiences, GAMMA,curr_invalid_choice,next_invalid_choice)

    def act(self, agent_state,invalid_choice,eps=0.):
        """
        Params
        ======
        :param state:(array_like) current state
        :param eps: epsilon, for epsilon-greedy action selection
        :return: return actions for given state per policy
        """

        # agent_state = torch.from_numpy(agent_state).float().unsqueeze(0).to(device)
        agent_state = torch.tensor(agent_state).float().unsqueeze(0)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(agent_state)
            action_values[0][list(invalid_choice.__args__[0])] = 0
        self.qnetwork_local.train()

        # epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            try:
                return random.choice(list(set(range(self.action_size))-invalid_choice.__args__[0]))
            except:
                return self.action_size
            # return random.choice(range(self.action_size))

    def learn(self, experiences, gamma,curr_invalid_choice,next_invalid_choice):
        """
       Params
       ======
       :param experiences:
       :param gamma:
       :return: return update value parameters using given batch experiences tuples
       """

        states, actions, rewards, next_states, dones = experiences

        # get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Q_targets_next[0][list(next_invalid_choice.__args__[0])] = 0
        # compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # get expected Q values from local model
        temp_Q_expected = self.qnetwork_local(states)
        Q_expected = torch.ones((32,1))
        for i,each in enumerate(actions):
            if each != self.action_size:
                Q_expected[i] = temp_Q_expected[i][each]
            else:
                Q_expected[i] = 0
        # Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Q_expected[0][list(curr_invalid_choice.__args__[0])] = 0
        # compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # minimize loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ---------------------------update target network---------------------#
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau * local_param.data + (1.0 - tau) * target_param.data)
