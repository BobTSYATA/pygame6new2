import pygame
import torch
from CONSTANTS import *
from Environment import Environment
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
from Human_Agent import Human_Agent
from Baseline_Agent import Baseline_Agent
from TroopUnit import TroopUnit
from State_ONLY_FOR_CALLING import State_ONLY_FOR_CALLING

class Tester:
    def __init__(self, env, player1, player2, main_surf) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2
        self.main_surf = main_surf
        

    def test(self, games_num):
        games = 0
        player_1 = self.player1  
        player_2 = self.player2
        environment = self.env


        while games < games_num:
            done = False
            print(f"Starting game {games + 1}")
            end_of_game = False

            environment.restart()
            after_state_2 = None
            state = environment.set_init_state(player_num="1")
            
            while not end_of_game:


                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        return
                events = None
                
                environment.draw_header(done, self.main_surf)
                # Player 1's turn
                if isinstance(self.player1, Random_Agent) or isinstance(self.player1, Baseline_Agent):
                    action_tuple_1 = player_1.get_Action(environment, "1", events=events)
                    reward1, player_1_done = environment.move(games_num,self.main_surf, action_tuple_1, agent_type="Random_Agent", player_num="1")
                elif isinstance(self.player1, DQN_Agent):
                    action_tuple_1 = player_1.get_Action(environment, state=state, events=events, train=False, epoch=games_num)
                    reward1, player_1_done = environment.move(games_num,self.main_surf, action_tuple_1, agent_type="DQN", player_num="1")

                after_state = environment.get_next_state(player_num="2")

                if player_1_done:
                    games += 1
                    end_of_game = True            
                    break


                # Player 2's turn (only after Player 1’s action is resolved)
                environment.draw_header(player_1_done, self.main_surf)
                if isinstance(self.player2, Random_Agent) or isinstance(self.player2, Baseline_Agent): 
                    action_tuple_2 = player_2.get_Action(environment, "2", events=events)
                    reward2, player_2_done = environment.move(games_num,self.main_surf, action_tuple_2, agent_type="Random_Agent", player_num="2")
                elif isinstance(self.player2, DQN_Agent):
                    action_tuple_2 = player_2.get_Action(environment, state=state, events=events, train=False, epoch=games_num)
                    reward2, player_2_done = environment.move(games_num,self.main_surf, action_tuple_2, agent_type="DQN", player_num="2")
                    print("action_tuple_2: ", action_tuple_2)
                
                # Check if Player 2 is done
                if player_2_done:
                    games += 1
                    end_of_game = True
                    break
                after_state_2 = environment.get_next_state(player_num="2") 
                state = after_state_2


            # Print game results
            print("Game over!")
            print(f"Player 1 Wins: {environment.player_1_won}, Player 2 Wins: {environment.player_2_won}")
            print(f"Games Played: {games}/{games_num}")

        print(f"Final Results: Player 1 Wins: {environment.player_1_won}, Player 2 Wins: {environment.player_2_won}")
        return environment.player_1_won, environment.player_2_won


    def __call__(self, games_num):
        return self.test(games_num)


# if __name__ == '__main__':
#     RUN_NUM = 125#107#102 is the best, ran for 100k smoothest graph. 107 ran for 50k but still got 94-6 so still good.
#     path = f"DataTraining/checkpoint{RUN_NUM}.pth" 

#     env = Environment()
#     main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
#     main_surf.fill(LIGHTGRAY) 
#     player1 = Random_Agent()#DQN_Agent(train=False,parametes_path=path,player_num=1)
#     player2 = DQN_Agent(train=False,parametes_path=path,player_num=2)#Random_Agent()
#     test = Tester(env,player1, player2,main_surf)
#     print(test.test(100))