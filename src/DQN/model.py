import torch
import torch.nn as nn
import torch.nn.functional as F


class QNetwork(nn.Module):

    def __init__(self,state_size,action_size,seed,fc1_units=64,fc2_units=32):
        super(QNetwork,self).__init__()
        self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size,fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units,action_size)

    # def forward(self,state,valid_actions):
    #     x = F.relu(self.fc1(state))
    #     x = F.relu(self.fc2(x))
    #     x = self.fc3(x)
    #     mask = torch.zeros_like(x)
    #     mask[valid_actions] = 1
    #     x = x * mask / sum(x * mask)

    def forward(self, state,invalid_choice):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x[0][list(invalid_choice.__args__[0])] = -10000
        x = F.softmax(x,dim=1)
        return x
