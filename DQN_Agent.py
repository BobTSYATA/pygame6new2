import torch
import random
import math
from DQN import DQN
from CONSTANTS import *
import numpy as np
from Environment import Environment
from State_ONLY_FOR_CALLING import State_ONLY_FOR_CALLING

epsilon_start = 1.0  # Starting epsilon value (usually high)
epsilon_final = 0.01#0.1  # Minimum epsilon value (usually low)
epsiln_decay = 3000#2000#5000#20000#1000  # Decay rate (number of episodes for decay)

class DQN_Agent:
    def __init__(self, parametes_path = None, train = True, env= None, player_num = 1):
        self.DQN = DQN()
        if parametes_path:
            self.DQN.load_params(parametes_path)
        #self.train = train
        self.setTrainMode(train)
        self.player = player_num
        self.env = env
        

    def setTrainMode (self, train):
          self.train = train
          if train:
              self.DQN.train()
          else:
              self.DQN.eval()

    def get_Action(self, environment, state : State_ONLY_FOR_CALLING, epoch=0, events=None, train=True) -> tuple: #train=False ##########

        # Get all possible actions for the agent
        #print("state.player: ",state.player)
        actions = environment.all_possible_actions_for_agent(state.player)
        if len(actions) == 0: ### to enable
            # return (None,) + (None,) ### to enable
            return ((-1,-1))
        epsilon = self.epsilon_greedy(epoch)
        rnd = random.random()

        actions_np = np.array(actions)

        actions_lst = actions_np.tolist()
        #print("actions_lst: ", actions_lst)
        if actions_lst == []:
            # return (None,) + (None,)
            return ((-1,-1))
        
        if train and rnd < epsilon:
            # Epsilon-greedy strategy
            #epsilon = self.epsilon_greedy(epoch)
            #if random.random() < epsilon:
            return random.choice(actions)  # Choose a random action from the list


        state_tensor = state.toTensor()

        with torch.no_grad():
            Q_values = self.DQN(state_tensor)#state  # Forward pass to get Q-values
            #print("11111111 Q_values: ", Q_values)
            # masking #
            #Q_values = Q_values * mask - (1 - mask) * 1e9
            #print("22222222 Q_values: ", Q_values)
            # masking #

        #print("Q_values: ", Q_values)
        # Select the action corresponding to the maximum Q-value
        max_index = torch.argmax(Q_values) #torch.argmax(Q_values[:len(actions)]).item()  # Convert tensor to Python integer

        if actions == []:
            # return (None,) + (None,) 
            return ((-1,-1))
        
        #print("actions[max_index]: ", actions[max_index])
        return actions[max_index]
    
    # def get_Actions_Values (self, states):
    #     with torch.no_grad():
    #         Q_values = self.DQN(states)
    #         max_values, max_indices = torch.max(Q_values,dim=1) # best_values, best_actions
    #     #print("max_indices.reshape(-1,1), max_values.reshape(-1,1): ",max_indices.reshape(-1,1), max_values.reshape(-1,1))
    #     return max_indices.reshape(-1,1), max_values.reshape(-1,1)

    # @staticmethod
    # def tensorToState (state_tensor, actions_tensor ,player):
    #     #print(f"state_tensor shape: {state_tensor.shape} state_tensor: {state_tensor}, actions_tensor shape: {actions_tensor.shape}, actions_tensor: {actions_tensor} ,player: {player}")
    #     board = state_tensor.reshape([1,136]).cpu().numpy()
    #     actions = actions_tensor.reshape([1,2]).cpu().numpy()
    #     actions = list(map(list, actions))
    #     return State_ONLY_FOR_CALLING(board, actions, player)


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
                actions.append((-1, -1))
                #continue
            else:
                actions.append(self.get_Action(environment,State_ONLY_FOR_CALLING.tensorToState(boards_tensor[i], actions_tensor[i],player = self.player), train=True)) #SARSA = True / Q-learning = False
        #print("actions_tensor: ", actions_tensor)
        actions_tensor = torch.tensor(actions).reshape(-1, 2)
        #  # actions_tensor = torch.tensor(actions).reshape(-1, 4)
        # # print()
        # # actions_np = np.array(actions).reshape(128,2)
        # # print(actions_np)
        # actions_tensor = torch.tensor(actions)
        # # actions_tensor = torch.from_numpy(actions_np)
        # return actions_tensor
        return actions_tensor


    def epsilon_greedy(self, epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay): # not defined: epsilon_start, epsilon_final, epsiln_decay
        res = final + (start - final) * math.exp(-1 * epoch/decay)
        return res
        # if epoch < epsiln_decay:
        #     return epsilon_start - (epsilon_start - epsilon_final) * epoch / epsiln_decay
        # return epsilon_final
        
    def loadModel (self, file):
        self.model = torch.load(file)


    def Q(self, states, actions): # problem might be here
        # Get Q-values from the DQN model
        Q_values = self.DQN(states)
        
        # Get the number of islands (N), which is needed to transform the (x, y) pairs
        N = 4  # For example, if you have 5 islands, set N = 5
        
        # Transform the actions (x, y) into action IDs
        # Assuming actions are a tensor of shape [batch_size, 2]
        action_ids = actions[:, 0] * N + actions[:, 1]  # Flatten the (x, y) into a unique action ID
        
        # Now action_ids is a 1D tensor of size [batch_size], containing the transformed action IDs
        #print(f"Transformed action IDs: {action_ids}")
        
        # Now, gather Q-values corresponding to the action IDs
        # We use .gather to index Q_values with action_ids
        # Q_values has shape [batch_size, num_actions] and action_ids is a vector of length [batch_size]
        Q_selected = Q_values.gather(1, action_ids.long().unsqueeze(1))  # .unsqueeze(1) reshapes to match dimensions
    
        return Q_selected
    def fix_update (self, dqn, tau=0.001):
        self.DQN.load_state_dict(dqn.state_dict())
    
    def save_param (self, path):
        self.DQN.save_params(path)

    def load_params (self, path):
        self.DQN.load_params(path)

    def __call__(self, events= None, state=None):
        return self.get_Action(state)
    

        # def get_actions(self, states, dones):
    #     actions = []
    #     print(f"dones: {dones}")
    #     for i, state in enumerate(states):
    #         if dones[i].item():  # If the episode is done for this state
    #             actions.append((None, None))  # Placeholder for done episode
    #         else:
    #             # Get all possible actions for the agent (use all_possible_actions_for_agent)
    #             all_possible_actions = self.env.all_possible_actions_for_agent(self.player)
                
    #             # Convert state to a tensor (this assumes you already have a state tensor)
    #             state_tensor = torch.tensor(state).unsqueeze(0)  # Add batch dimension if needed
                
    #             # Get Q-values for all possible actions
    #             with torch.no_grad():
    #                 Q_values = self.DQN(state_tensor)  # Get Q-values for the current state
                
    #             # Select the action with the highest Q-value
    #             max_action_index = torch.argmax(Q_values).item()
    #             action = all_possible_actions[max_action_index]
                
    #             actions.append(action)  # Store the chosen action

    #     # Convert actions list into a tensor (or keep it as a list of tuples)
    #     actions_tensor = torch.tensor(actions)
    #     print("actions_tensor: ",actions_tensor)
    #     return actions_tensor




        # def Q (self, states, actions):
    #     Q_values = self.DQN(states) # try: Q_values = self.DQN(states).gather(dim=1, actions) ; check if shape of actions is [-1, 1] otherwise dim=0
    #     print("Q_values: ", Q_values)
    #     rows = torch.arange(Q_values.shape[0]).reshape(-1,1)
    #     cols = actions.long().reshape(-1,1)
    #     print(f"Q_values shape: {Q_values.shape}")  # Should be [batch_size, num_actions]
    #     print(f"actions shape: {actions.shape}")    # Should be [batch_size, 1]
    #     print("actions: ", actions)
    #     return Q_values[rows, cols]












# import torch
# import random
# import math
# from DQN import DQN
# from CONSTANTS import *

# epsilon_start = 1.0  # Starting epsilon value (usually high)
# epsilon_final = 0.01#0.1  # Minimum epsilon value (usually low)
# epsiln_decay = 5000#20000#1000  # Decay rate (number of episodes for decay)

# class DQN_Agent:
#     def __init__(self, parametes_path = None, train = True, env= None):
#         self.DQN = DQN()
#         if parametes_path:
#             self.DQN.load_params(parametes_path)
#         self.train = train
#         self.setTrainMode()
        

#     def setTrainMode (self):
#           if self.train:
#               self.DQN.train()
#           else:
#               self.DQN.eval()

#     def get_Action(self, environment, player_num, events=None, state=None, epoch=0, train=True) -> tuple: #train=False ##########
#         agent_type = 1 if player_num == "1" else 2

#         # Get all possible actions for the agent
#         actions = environment.all_possible_actions_for_agent(agent_type)
#         if len(actions) == 0: ### to enable
#             return (None,) + (None,) ### to enable

#         # implementing masking from wiki:Masking involves creating a mask that disables invalid actions by assigning very low Q-values to them. This way, even if the action space size is fixed, the agent will not choose invalid actions.#
#         num_actions = len(actions)
#         # mask = torch.zeros(64, dtype=torch.float32)  # Mask for all possible actions # 64 = max output_size
#         # mask[:num_actions] = 1  # Enable only valid actions
#         # masking #

#         if self.train and train:
#             # Epsilon-greedy strategy
#             epsilon = self.epsilon_greedy(epoch)
#             if random.random() < epsilon:
#                 return random.choice(actions)  # Choose a random action from the list

#         with torch.no_grad():
#             Q_values = self.DQN(state)  # Forward pass to get Q-values
#             print("11111111 Q_values: ", Q_values)
#             # masking #
#             #Q_values = Q_values * mask - (1 - mask) * 1e9
#             print("22222222 Q_values: ", Q_values)
#             # masking #

#         # masking #
#         #max_index = torch.argmax(Q_values).item() # gets the int from the tensor
#         # masking #
#         print("Q_values: ", Q_values)
#         # Select the action corresponding to the maximum Q-value
#         max_index = torch.argmax(Q_values) #torch.argmax(Q_values[:len(actions)]).item()  # Convert tensor to Python integer
#         # print("max_index: ", max_index)
#         # print("len(actions): ", len(actions), " actions: ", actions)

#         if len(actions) == 0: ### to enable
#             return (None,) + (None,) ### to enable
        
#         print("actions[max_index]: ", actions[max_index])
#         return actions[max_index]

#     def epsilon_greedy(self, epoch, start = epsilon_start, final=epsilon_final, decay=epsiln_decay): # not defined: epsilon_start, epsilon_final, epsiln_decay
#         res = final + (start - final) * math.exp(-1 * epoch/decay)
#         return res
#         # if epoch < epsiln_decay:
#         #     return epsilon_start - (epsilon_start - epsilon_final) * epoch / epsiln_decay
#         # return epsilon_final
        
#     def loadModel (self, file):
#         self.model = torch.load(file)
    
#     def save_param (self, path):
#         self.DQN.save_params(path)

#     def load_params (self, path):
#         self.DQN.load_params(path)

#     def __call__(self, events= None, state=None):
#         return self.get_Action(state)











    # def get_Actions_Values (self, states): ##########
    #     with torch.no_grad():
    #         Q_values = self.DQN(states)
    #         max_values, max_indices = torch.max(Q_values,dim=1) # best_values, best_actions
        
    #     return max_indices.reshape(-1,1), max_values.reshape(-1,1)


    # def Q (self, states, actions):################## # TO REDO LIKE GILAD TOLD
    #     Q_values = self.DQN(states)
    #     rows = torch.arange(Q_values.shape[0]).reshape(-1,1)
    #     #print("actions.reshape(-1,1): ", actions.reshape(-1,1), " actions: ", actions)

    #     actions_int = []
    #     for action in actions:
    #         # Combine two integers from the tensor into a single integer
    #         #combined_int = int(f"{int(action[0])}{int(action[1])}")
    #         combined_int = action[1] #action[0] * 8 + action[1] # TO REDO LIKE GILAD TOLD
    #         #print("action: ", action, " combined_int: ", combined_int)
    #         actions_int.append(combined_int)

    #     #cols = actions_int.reshape(-1,1) #actions #actions.reshape(-1,1)
    #     cols = torch.tensor(actions_int, dtype=torch.long).reshape(-1, 1)

    #     # Print debugging information
    #     # print("actions: ", actions)
    #     # print("rows: ", rows)
    #     # print("cols: ", cols)
    #     # print("len(rows): ", len(rows))
    #     # print("len(cols): ", len(cols))
    #     # print("Q_values length: ", len(Q_values))
    #     # print("Q_values shape[1]:", Q_values.shape[1])  # This should be 1
    #     # print("Maximum cols:", max(cols))  # This should be <= 7*8 (i.e., 63 for 8x8 grid)

    #     return Q_values[rows, cols]






























































    # def Q(self, states, actions): # to try do like reversi?
    #     # Get the Q-values from the model
    #     Q_values = self.DQN(states)  # shape: [batch_size, 512]
    #     #print(f"Q_values shape: {Q_values.shape}")  # Should be [50, 512]

    #     # If actions has shape [50, 9], you need to reduce it to [50]
    #     # You can either take the index of the action or select one action
    #     actions = actions[:, 0]  # Take the first action (or any other logic to select one action)

    #     #print(f"Actions shape after reduction: {actions.shape}")  # Should be [50]

    #     # Ensure actions is a 1D tensor of shape [batch_size] (e.g., [50])
    #     if len(actions.shape) > 1:
    #         actions = actions.squeeze(1)  # Convert actions to shape [batch_size] (e.g., [50])

    #     #print(f"Actions shape: {actions.shape}")  # Should be [50]

    #     # Gather the Q-values corresponding to the chosen actions
    #     print("actions.unsqueeze(1): ", actions.unsqueeze(1))
    #     selected_Q_values = Q_values.gather(1, actions.unsqueeze(1))  # shape: [batch_size, 1]
    #     #print(f"Selected Q-values shape: {selected_Q_values.shape}")  # Should be [50, 1]

    #     return selected_Q_values