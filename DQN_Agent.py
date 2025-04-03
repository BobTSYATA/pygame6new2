import torch
import random
import math
from DQN import DQN
from CONSTANTS import *
import numpy as np
from Environment import Environment
from State_ONLY_FOR_CALLING import State_ONLY_FOR_CALLING

epsilon_start = 1
epsilon_final = 0.01#0.05
epsiln_decay = 5000# 2000

class DQN_Agent:
    def __init__(self, parametes_path = None, train = True, env= None, player_num = 1):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        self.setTrainMode(train)
        self.player = player_num
        self.env = env
        

    def setTrainMode (self, train):
          self.train = train
          if train:
              self.DQN.train()
          else:
              self.DQN.eval()


    def get_Action(self, environment, state : State_ONLY_FOR_CALLING, epoch=0, events=None, train=False) -> tuple:

        actions = environment.all_possible_actions_for_agent(self.player)
        if len(actions) == 0:
            return ((-1,-1))

        epsilon = self.epsilon_greedy(epoch)
        rnd = random.random()

        actions_np = np.array(actions)

        actions_lst = actions_np.tolist()
        if actions_lst == []:
            print("actions_lst == [] TRUE: ((-1,-1))")
            return ((-1,-1))
        
        if train and rnd < epsilon:
            return random.choice(actions) 


        state_tensor = state.toTensor()

        action_tensor = torch.tensor(actions).reshape(-1, 2)
        expand_state_tensor = state_tensor[0].unsqueeze(0).repeat((len(action_tensor),1))

        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, action_tensor)

        max_index = torch.argmax(Q_values) 

        if actions == []:

            print("actions == [] TRUE: ((-1,-1))")
            return ((-1,-1))

        return actions[max_index]



    def get_actions (self, environment, states, dones):
        actions = []
        boards_tensor = states[0]
        actions_tensor = states[1]

        for i, state in enumerate(boards_tensor):
            if dones[i].item():
                actions.append((-1, -1))
            else:
                actions.append(self.get_Action(environment,State_ONLY_FOR_CALLING.tensorToState(environment, boards_tensor[i], actions_tensor[i]), train=True))
        actions_tensor = torch.tensor(actions).reshape(-1, 2)
        return actions_tensor


    def epsilon_greedy(self, epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay):
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
        
    def loadModel (self, file):
        self.model = torch.load(file)


    def fix_update (self, dqn, tau=0.001):
        self.DQN.load_state_dict(dqn.state_dict())
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)