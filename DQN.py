import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from CONSTANTS import *

# Parameters
input_size = STATE_SIZE + 2 # Q(s,a)
layer1 = 64
layer2 = 32
output_size = 1 
gamma = 0.75#0.6#0.85#0.99
MSELoss = nn.MSELoss()

class DQN(nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        self.output = nn.Linear(layer2, output_size) 


    def forward (self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, negative_slope=0.01)#F.relu(x) 
        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.01)#F.relu(x)
        x = self.output(x)
        return x
    
    def loss (self, Q_values, rewards, Q_next_Values, dones):#, dones
        Q_target = rewards + gamma * Q_next_Values * (1 - dones)
        return MSELoss(Q_values, Q_target)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self): 
        new_DQN = DQN()
        new_DQN.load_state_dict(self.state_dict())
        return new_DQN

    def __call__(self, states, actions):
        state_action = torch.cat((states,actions), dim=1)
        return self.forward(state_action)
