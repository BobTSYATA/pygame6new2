from collections import deque
import random
import torch
import numpy as np
from State_ONLY_FOR_CALLING import State_ONLY_FOR_CALLING

capacity = 100000#500000

class ReplayBuffer:
    def __init__(self, capacity= capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    # def push (self, state , action, reward, next_state, done): # , done
    #     state = torch.tensor(state, dtype=torch.float32) if not isinstance(state, torch.Tensor) else state
    #     action = torch.tensor(action, dtype=torch.float32).unsqueeze(0) if not isinstance(action, torch.Tensor) else action
    #     reward = torch.tensor([reward], dtype=torch.float32)  # Make sure it's a 1D tensor
    #     next_state = torch.tensor(next_state, dtype=torch.float32) if not isinstance(next_state, torch.Tensor) else next_state
    #     done = torch.tensor([done], dtype=torch.float32)  # Make sure done is also a 1D tensor

        # if not isinstance(reward, torch.Tensor):
        #     print("reward is NOT a tensor!")
        # else:
        #     print("reward is a tensor!")

        # if isinstance(action, np.ndarray):
        #     print("action is a NumPy array!")
        # else:
        #     print("action is NOT a NumPy array!")

    #     self.buffer.append((state, action, reward, next_state, done))# idk how i should pass this on maybe tensor and numpy shaped / floats and ints, etc. TO DO: to check if they are tensors / np.array and if not to do like in checkers example # , done
    def push (self, state : State_ONLY_FOR_CALLING, action, reward, next_state : State_ONLY_FOR_CALLING, done): # at Trainer_wandb i don't give it a State
        #print("state: ", state)
        self.buffer.append((state.toTensor(), torch.from_numpy(np.array(action).reshape(-1,2)), torch.tensor(reward), next_state.toTensor(), torch.tensor(done)))



    def sample (self, batch_size):
        if (batch_size > self.__len__()):
            batch_size = self.__len__()
        state_tensors, action_tensor, reward_tensors, next_state_tensors, dones = zip(*random.sample(self.buffer, batch_size))
        state_boards, state_actions = zip(*state_tensors)
        states = torch.vstack(state_boards), state_actions
        actions = torch.vstack(action_tensor)
        rewards = torch.vstack(reward_tensors)
        next_board, next_actions = zip(*next_state_tensors)
        next_states = torch.vstack(next_board), next_actions
        done_tensor = torch.tensor(dones).long().reshape(-1,1)
        return states, actions, rewards, next_states, done_tensor
    # def sample (self, batch_size):
    #     if (batch_size > self.__len__()):
    #         batch_size = self.__len__()
    #     #state_tensors, action_tensor, reward_tensors, next_state_tensors = zip(*random.sample(self.buffer, batch_size)) # next_state_tensors , dones_tensor
    #     # print("batch_size: ", batch_size, " self.buffer: ", self.buffer)

    #     state_tensors, action_tensor, reward_tensors, next_state_tensors, dones_tensor = zip(*random.sample(self.buffer, batch_size))
    #     #print(f"len(state_tensors[0]): {len(state_tensors[0])}, len(state_tensors[0]): {len(action_tensor[0])}, reward_tensors[0]: {reward_tensors[0]}, len(next_state_tensors[0]): {len(next_state_tensors[0])}, dones_tensor[0]: {dones_tensor[0]} ")
    #     if isinstance(state_tensors, (tuple, list)):
    #         print("GOING TO UNPACK?")  # Unpack only if it's iterable
    #         print("state_tensors: ", state_tensors, "state_tensors[0]: ", state_tensors[0])
    #         state_boards, state_actions = zip(*state_tensors)
    #     else:
    #         print("state_tensors is not iterable, checking structure")
    #         print(state_tensors)
    #     # print(f" state_tensors[0].shape: {state_tensors[0].shape} , state_tensors: {state_tensors[0]}")
    #     # for i, item in enumerate(state_tensors[0]):
    #     #     if item.dim() > 0:  # Only check length if it's not a 0-dimensional tensor
    #     #         print(f"state_tensors[{i}] has {len(item)} elements: {item}")
    #     #     else:
    #     #         print(f"state_tensors[{i}] is a 0-dimensional tensor with value: {item.item()}")
    #     state_boards, state_actions = zip(*state_tensors)
    #     #print("player: ", player)
    #     #state_boards, state_actions, *_ = zip(*state_tensors)  # Unpacks first two, ignores extra
    #     states = torch.vstack(state_boards), state_actions#torch.vstack(state_tensors)
    #     actions= torch.vstack(action_tensor)
    #     rewards = torch.vstack(reward_tensors)
    #     next_board, next_actions = zip(*next_state_tensors)
    #     next_states = torch.vstack(next_board), next_actions
    #     dones = torch.vstack(dones_tensor).long().reshape(-1,1)
    #     return states, actions, rewards, next_states, dones
    #     # state_tensors, action_tensor, reward_tensors, next_state_tensors, dones = zip(*random.sample(self.buffer, batch_size)) # gives the error: ValueError: too many values to unpack (expected 2)
    #     # state_boards, state_actions = zip(*state_tensors) 
    #     # states = torch.vstack(state_boards), state_actions
    #     # #states = torch.vstack(state_tensors)
    #     # actions= torch.vstack(action_tensor)
    #     # rewards = torch.vstack(reward_tensors)
    #     # next_board, next_actions = zip(*next_state_tensors)
    #     # next_states = torch.vstack(next_board), next_actions #torch.vstack(next_state_tensors)
    #     # dones = torch.tensor(dones).long().reshape(-1,1)#torch.vstack(dones_tensor)
    #     # #print("states, actions, rewards, next_states: ", states, actions, rewards, next_states)
    #     # return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
    






# from collections import deque
# import random
# import torch
# import numpy as np

# capacity = 100000#500000

# class ReplayBuffer:
#     def __init__(self, capacity= capacity, path = None) -> None:
#         if path:
#             self.buffer = torch.load(path).buffer
#         else:
#             self.buffer = deque(maxlen=capacity)

#     def push (self, state , action, reward, next_state, done): # , done
#         self.buffer.append((state, action, reward, next_state, done)) # , done
    
#     def sample (self, batch_size):
#         if (batch_size > self.__len__()):
#             batch_size = self.__len__()
#         #state_tensors, action_tensor, reward_tensors, next_state_tensors = zip(*random.sample(self.buffer, batch_size)) # next_state_tensors , dones_tensor
#         state_tensors, action_tensor, reward_tensors, next_state_tensors, dones_tensor = zip(*random.sample(self.buffer, batch_size))
#         ### to do: save everything and redo this part from ReplayBuffer.py in reversi. ##
#         states = torch.vstack(state_tensors)
#         actions= torch.vstack(action_tensor)
#         rewards = torch.vstack(reward_tensors)
#         next_states = torch.vstack(next_state_tensors)
#         dones = torch.vstack(dones_tensor)
#         #print("states, actions, rewards, next_states: ", states, actions, rewards, next_states)
#         return states, actions, rewards, next_states, dones

#     def __len__(self):
#         return len(self.buffer)