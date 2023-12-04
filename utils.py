import torch.nn as nn
import torch.nn.functional as F


def log_params(filename, params_dict):
    f = open('./logs/log_' + filename + '.txt', 'w')
    for key, value in params_dict.items():
        f.write(key + f': {str(value)}' + '\n')
    f.close()


class DQN(nn.Module):

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        return self.layer3(x)
