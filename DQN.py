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
output_size = 16#64#8 #9  # Q(state) -> 4 value of stay, left, right, shoot
gamma = 0.99#0.85#0.99
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
        x = F.relu(x) # leaky_relu
        x = self.linear2(x)
        x = F.relu(x)
        # x = self.linear3(x)
        # x = F.leaky_relu(x)
        x = self.output(x)
        #print("x: ", x, " len(x): ", len(x))
        return x
    
    def loss (self, Q_values, rewards, Q_next_Values, dones):#, dones
        Q_new = rewards + gamma * Q_next_Values * (1 - dones) # Q_next_Values.detach()
        return MSELoss(Q_values, Q_new)
    
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
















    # def __call__(self, states, actions):
    #     #states = states.unsqueeze(0)  # Convert [136] -> [1, 136]
    #     print("states shape: ", states.shape)
    #     print("actions shape: ", actions.shape)
    #     state_action = torch.cat((states,actions), dim=1) # 
    #     #print("state_action: ", state_action)
    #     return self.forward(state_action)
        
    #     # return self.forward(states)








# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import copy
# from CONSTANTS import *

# # Parameters
# print("STATE SIZE: ", STATE_SIZE)
# input_size = STATE_SIZE #152 for 8 islands, 136 for 4 islands  # Q(state) see environment for state shape
# layer1 = 64#322
# layer2 = 32#468
# #layer3 = 64
# # layer4 = 64  # Adjusted to match the output of linear5
# # layer5 = 64
# output_size = 16#64#8 #9  # Q(state) -> 4 value of stay, left, right, shoot
# gamma = 0.85#0.99

# class DQN(nn.Module):
#     def __init__(self, device = torch.device('cpu')) -> None:
#         super().__init__()
#         self.device = device
#         self.linear1 = nn.Linear(input_size, layer1)
#         self.linear2 = nn.Linear(layer1, layer2)
#         #self.linear3 = nn.Linear(layer2, layer3)
#         # self.linear4 = nn.Linear(layer3, layer4)  # Now outputs 64
#         # self.linear5 = nn.Linear(layer4, layer5)  # Output size is 64
#         self.output = nn.Linear(layer2, output_size)  # Output layer remains the same
#         self.MSELoss = nn.MSELoss()

#     def forward (self, x):
#         x = self.linear1(x)
#         x = F.leaky_relu(x)
#         x = self.linear2(x)
#         x = F.leaky_relu(x)
#         # x = self.linear3(x)
#         # x = F.leaky_relu(x)
#         x = self.output(x)
#         #print("x: ", x, " len(x): ", len(x))
#         return x
    
#     def loss (self, Q_values, rewards, Q_next_Values, dones):#, dones
#         Q_new = rewards + gamma * Q_next_Values * (1 - dones)
#         return self.MSELoss(Q_values, Q_new)
    
#     def load_params(self, path):
#         self.load_state_dict(torch.load(path))

#     def save_params(self, path):
#         torch.save(self.state_dict(), path)

#     def copy (self): ###########
#         return copy.deepcopy(self)

#     def __call__(self, states):
#         return self.forward(states)
