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


    def get_Action(self, environment, state : State_ONLY_FOR_CALLING, epoch=0, events=None, train=False) -> tuple: #train=False ##########

        # Get all possible actions for the agent
        #print("state.player: ",state.player)
        # print("calling all_possible_actions_for_agent from DQN_Agent get_Action")
        actions = environment.all_possible_actions_for_agent(self.player)#state.player)
        if len(actions) == 0: ### to enable
            # return (None,) + (None,) ### to enable
            # print("len(actions) == 0 TRUE: ((-1,-1))")
            return ((-1,-1))
        # print("len(actions) == 0 FALSE")
        # print("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        epsilon = self.epsilon_greedy(epoch)
        # print("epsilon: ", epsilon)
        rnd = random.random()

        actions_np = np.array(actions)

        actions_lst = actions_np.tolist()
        #print("actions_lst: ", actions_lst)
        if actions_lst == []:
            # return (None,) + (None,)
            print("actions_lst == [] TRUE: ((-1,-1))")
            return ((-1,-1))
        
        if train and rnd < epsilon:
            # print("random.choice(actions)")
            return random.choice(actions)  # Choose a random action from the list


        state_tensor = state.toTensor()

        action_tensor = torch.tensor(actions).reshape(-1, 2)
        expand_state_tensor = state_tensor[0].unsqueeze(0).repeat((len(action_tensor),1))

        # print(f"action_tensor: {action_tensor}, expand_state_tensor: {expand_state_tensor}")

        with torch.no_grad():
            Q_values = self.DQN(expand_state_tensor, action_tensor)#state  # Forward pass to get Q-values


        max_index = torch.argmax(Q_values) #torch.argmax(Q_values[:len(actions)]).item()  # Convert tensor to Python integer

        if actions == []:
            # return (None,) + (None,) 
            print("actions == [] TRUE: ((-1,-1))")
            return ((-1,-1))
        
        #print("actions[max_index]: ", actions[max_index])
        # print("actions[max_index]")
        return actions[max_index]



    def get_actions (self, environment, states, dones):
        actions = []
        #print("states: ", states)
        boards_tensor = states[0]
        actions_tensor = states[1]
        #print("len(actions_tensor): ", len(actions_tensor))
        for i, state in enumerate(boards_tensor):
            if dones[i].item():
                #print("(None,) + (None,): ", (None,) + (None,))
                #actions.append((None,) + (None,))
                # print("end of game so appending (-1,-1)")
                actions.append((-1, -1))
                #continue
            else:
                actions.append(self.get_Action(environment,State_ONLY_FOR_CALLING.tensorToState(environment, boards_tensor[i], actions_tensor[i]), train=True))#,player = self.player #SARSA = True / Q-learning = False
        #print("actions_tensor: ", actions_tensor)
        actions_tensor = torch.tensor(actions).reshape(-1, 2)
        return actions_tensor


    def epsilon_greedy(self, epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay): # not defined: epsilon_start, epsilon_final, epsiln_decay
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