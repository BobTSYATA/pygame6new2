from CONSTANTS import *
import pygame
import numpy as np
import torch

class State_ONLY_FOR_CALLING:
    def __init__(self, environment = None, Graphics = None, legal_actions = []):# ,player = 1):
        self.environment = environment
        if Graphics is not None:
            self.Graphics = Graphics
        else:
            self.Graphics = self.init_state(environment)    
       # self.player = player 
        self.legal_actions = legal_actions


    def toTensor (self, device = torch.device('cpu')):
        # if self.Graphics == 
        board_np = self.Graphics.reshape(-1)
        board_tensor = torch.tensor(board_np, dtype=torch.float32, device=device)
        actions_np = np.array(self.legal_actions)
        actions_tensor = torch.from_numpy(actions_np)
        #print(f"board_tensor: {board_tensor}, actions_tensor: {actions_tensor}")
        return board_tensor, actions_tensor

    def tensorToState (environment, state_tensor, actions_tensor):# ,player = 1): 
        board = state_tensor.reshape([1,STATE_SIZE]).cpu().numpy() 
        #print("actions_tensor: ", actions_tensor)
        actions = actions_tensor.reshape([-1,2]).cpu().numpy()
        actions = list(map(list, actions))
        #print("board: ", board, " actions: ",actions, " player: ", player)
        return State_ONLY_FOR_CALLING(environment=environment,Graphics=board, legal_actions=actions)#, player)


    def initialize_state_tensor(self):
        return torch.zeros(STATE_SIZE)

    def init_state(self, env):
        # Initialize state tensor
        state_tensor = torch.zeros(STATE_SIZE)#self.initialize_state_tensor()

        # DISABLED BECAUSE MADE IT INTO A STEP BASED GAME #
        # # Number of elements per troop unit
        # ELEMENTS_PER_TROOP_UNIT = 6
        # # print("len(env.troop_units): ", len(env.troop_units))
        # # Fill in the troop units data
        # for i, troop_unit in enumerate(env.troop_units[:MAX_TROOP_UNITS]):
        #     # print("troop_unit.type: ", troop_unit.type)
        #     index = i * ELEMENTS_PER_TROOP_UNIT
        #     state_tensor[index] = troop_unit.rect.centerx
        #     state_tensor[index + 1] = troop_unit.rect.centery
        #     state_tensor[index + 2] = troop_unit.troop_count
        #     state_tensor[index + 3] = troop_unit.type  # R
        #     state_tensor[index + 4] = troop_unit.destination[0]  # x-coordinate of destination
        #     state_tensor[index + 5] = troop_unit.destination[1]  # y-coordinate of destination

        # Number of elements per island
        ELEMENTS_PER_ISLAND = 4

        # Start index for islands data
        island_start_index = 0 # MAX_TROOP_UNITS * ELEMENTS_PER_TROOP_UNIT
        # print("")
        # Fill in the islands data
        for i, island in enumerate(env.islands[:MAX_ISLANDS]):
            index = island_start_index + i * ELEMENTS_PER_ISLAND
            state_tensor[index] = island.x
            state_tensor[index + 1] = island.y
            state_tensor[index + 2] = island.type  # Assuming type is a numerical representation
            state_tensor[index + 3] = island.troops

        #print("state_tensor: ", state_tensor)

        return state_tensor