import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from CONSTANTS import *

# Parameters
#print("STATE SIZE: ", STATE_SIZE)
###### CHANGING STRUCTURE ######
input_size = STATE_SIZE + 2# = 136 + 2 (OF THE ACTION) #152 for 8 islands, 136 for 4 islands  # Q(state) see environment for state shape
###### CHANGING STRUCTURE ######
layer1 = 64#322
layer2 = 32#468
#layer3 = 64
# layer4 = 64  # Adjusted to match the output of linear5
# layer5 = 64
output_size = 1 # Q(s,a) #16#64#8 #9  # Q(state) -> 4 value of stay, left, right, shoot
gamma = 0.75#0.6#0.85#0.99
MSELoss = nn.MSELoss()

class DQN(nn.Module):
    def __init__(self, device = torch.device('cpu')) -> None:
        super().__init__()
        self.device = device
        self.linear1 = nn.Linear(input_size, layer1)
        self.linear2 = nn.Linear(layer1, layer2)
        #self.linear3 = nn.Linear(layer2, layer3)
        # self.linear4 = nn.Linear(layer3, layer4)  # Now outputs 64
        # self.linear5 = nn.Linear(layer4, layer5)  # Output size is 64
        self.output = nn.Linear(layer2, output_size)  # Output layer remains the same


    def forward (self, x):
        x = self.linear1(x)
        x = F.leaky_relu(x, negative_slope=0.01)#F.relu(x) # leaky_relu
        x = self.linear2(x)
        x = F.leaky_relu(x, negative_slope=0.01)#F.relu(x)
        # x = self.linear3(x)
        # x = F.leaky_relu(x)
        x = self.output(x)
        #print("x: ", x, " len(x): ", len(x))
        # print("x: ", x)
        # print(f"222222222222222 Forward pass output Q-values: {x[:5].detach().cpu().numpy()}")
        return x
    
    def loss (self, Q_values, rewards, Q_next_Values, dones):#, dones
        Q_target = rewards + gamma * Q_next_Values * (1 - dones)#Q_next_Values.max(dim=1, keepdim=True)[0].detach() #Q_target = Q_new #Q_next_Values.max(1)[0].detach() # Q_next_Values.detach()
        # print(f"Loss Debug - Q_values: {Q_values[:5].detach().cpu().numpy()},")
        # print(f"Loss Debug - Q_target (target values): {Q_target[:5].detach().cpu().numpy()}")
        return MSELoss(Q_values, Q_target)
    
    def load_params(self, path):
        self.load_state_dict(torch.load(path))

    def save_params(self, path):
        torch.save(self.state_dict(), path)

    def copy (self): ###########
        new_DQN = DQN()
        new_DQN.load_state_dict(self.state_dict())
        return new_DQN
        #return copy.deepcopy(self)

    def __call__(self, states, actions):
        #print("states: ", states.shape, " actions: ", actions.shape)
        state_action = torch.cat((states,actions), dim=1)
        return self.forward(state_action)

    ###### CHANGING STRUCTURE ######
    # def __call__(self, states):
    #     return self.forward(states)
    ###### CHANGING STRUCTURE ######