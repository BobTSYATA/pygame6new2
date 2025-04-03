from CONSTANTS import *
import pygame
import numpy as np
import torch

class State_ONLY_FOR_CALLING:
    def __init__(self, environment = None, Graphics = None, legal_actions = []):
        self.environment = environment
        if Graphics is not None:
            self.Graphics = Graphics
        else:
            self.Graphics = self.init_state(environment)    
        self.legal_actions = legal_actions


    def toTensor (self, device = torch.device('cpu')):
        board_np = self.Graphics.reshape(-1)
        board_tensor = torch.tensor(board_np, dtype=torch.float32, device=device)
        actions_np = np.array(self.legal_actions)
        actions_tensor = torch.from_numpy(actions_np)
        return board_tensor, actions_tensor

    def tensorToState (environment, state_tensor, actions_tensor):
        board = state_tensor.reshape([1,STATE_SIZE]).cpu().numpy() 
        actions = actions_tensor.reshape([-1,2]).cpu().numpy()
        actions = list(map(list, actions))
        return State_ONLY_FOR_CALLING(environment=environment,Graphics=board, legal_actions=actions)


    def initialize_state_tensor(self):
        return torch.zeros(STATE_SIZE)

    def init_state(self, env):
        # Initialize state tensor
        state_tensor = torch.zeros(STATE_SIZE)

        # Number of elements per island
        ELEMENTS_PER_ISLAND = 4

        # Start index for islands data
        island_start_index = 0 # MAX_TROOP_UNITS * ELEMENTS_PER_TROOP_UNIT

        # Fill in the islands data
        for i, island in enumerate(env.islands[:MAX_ISLANDS]):
            index = island_start_index + i * ELEMENTS_PER_ISLAND
            state_tensor[index] = island.x
            state_tensor[index + 1] = island.y
            state_tensor[index + 2] = island.type 
            state_tensor[index + 3] = island.troops

        return state_tensor