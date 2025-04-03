from collections import deque
import random
import torch
import numpy as np
from State_ONLY_FOR_CALLING import State_ONLY_FOR_CALLING

capacity = 100000

class ReplayBuffer:
    def __init__(self, capacity= capacity) -> None:
        self.buffer = deque(maxlen=capacity)

    def push (self, state : State_ONLY_FOR_CALLING, action, reward, next_state : State_ONLY_FOR_CALLING, done): 
        if not action == (-1,-1):
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
   
    def __len__(self):
        return len(self.buffer)
    