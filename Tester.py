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


class Tester:
    def __init__(self, env, player1, player2) -> None:
        self.env = env
        self.player1 = player1
        self.player2 = player2
        

    def test(self, games_num):
        games = 0
        player_1 = self.player1  # Use the initialized players
        player_2 = self.player2
        pygame.init()

        environment = self.env  # Use the initialized environment
        clock = pygame.time.Clock()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))  # Initialize the screen
        main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
        main_surf.fill(LIGHTGRAY)

        while games < games_num:
            end_of_game = False
            print(f"Starting game {games + 1}")
            
            # Reset environment for the next game
            environment.restart()
            environment.reset_troop_units()
            
            while not end_of_game:
                state = environment.state()
                
                # Handle events
                events = pygame.event.get()
                for event in events:
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return

                # Player 1's action
                action_tuple_1 = player_1.get_Action(environment, events=events, state=state)
                reward, player_1_done = environment.move(action_tuple_1, agent_type="DQN", player_num="1", state=state)

                # Update troops
                #environment.update_troop_units() # done in move

                # Check if Player 1 is done
                if player_1_done:
                    games += 1
                    end_of_game = True
                    
                    break

                # Player 2's action
                action_tuple_2 = player_2.get_Action(environment, player_num="2", events=events)
                reward, player_2_done = environment.move(action_tuple_2, agent_type="Random_Agent", player_num="2", state=state)

                # Check if Player 2 is done
                if player_2_done:
                    games += 1
                    end_of_game = True
                    print("dfg")
                    break

                # Draw everything
                environment.draw_header(False, main_surf)
                #screen.blit(main_surf, (0, 0))
                #pygame.display.flip()

            # Print game results
            print("Game over!")
            print(f"Player 1 Wins: {environment.player_1_won}, Player 2 Wins: {environment.player_2_won}")
            print(f"Games Played: {games}/{games_num}")

        pygame.quit()
        print(f"Final Results: Player 1 Wins: {environment.player_1_won}, Player 2 Wins: {environment.player_2_won}")
        return environment.player_1_won, environment.player_2_won


    def __call__(self, games_num):
        return self.test(games_num)


if __name__ == '__main__':
    RUN_NUM = 43
    path = f"DataTraining/checkpoint{RUN_NUM}.pth" 

    env = Environment()
    player1 = DQN_Agent(parametes_path=path, train=False, env=env)
    player2 = Random_Agent()
    test = Tester(env,player1, player2)
    print(test.test(100))



































# import pygame
# import torch
# from CONSTANTS import *
# from Environment import Environment
# from DQN_Agent import DQN_Agent
# from ReplayBuffer import ReplayBuffer
# from Random_Agent import Random_Agent
# from Human_Agent import Human_Agent
# from Baseline_Agent import Baseline_Agent
# from TroopUnit import TroopUnit


# class Tester:
#     def __init__(self, env, player1, player2) -> None:
#         self.env = env
#         self.player1 = player1
#         self.player2 = player2
        

#     def test (self, games_num):
      
#         games = 0
#         player_1 = DQN_Agent()
#         player_2 = Random_Agent()
#         # player_1 = self.player1
#         # player_2 = self.player2
#         # environment = Environment()
#         pygame.init()
#         environment = Environment()
#         clock = pygame.time.Clock()
#         main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
#         main_surf.fill(LIGHTGRAY)
#         while games < games_num:
#             end_of_game = False
#             print("weeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee")
#             while not end_of_game:
#                 state = environment.state()
 
#                 events = pygame.event.get()
#                 for event in events:
#                     if event.type == pygame.QUIT:
#                         return
                
                
    
#                 # Player 1's action
#                 action_tuple_1 = player1.get_Action(environment, player_num="1", events=events, state=state)
#                 player_done = environment.move(action_tuple_1, agent_type="DQN", player_num="1",state=state)

#                 conquering_troops = environment.update_troop_units()

#                 if player_done:
#                     games += 1
#                     break

#                 environment.draw_header(player_done, main_surf)
                
#                 #environment.all_possible_actions_for_agent(1)

#                 # Player 2's action
#                 action_tuple_2 = player2.get_Action(environment, "2", events)
#                 player_done_random = environment.move(action_tuple_2, agent_type="Random_Agent", player_num="2",state=state)


#                 if player_done_random:
#                     games += 1
#                     break
#                 environment.draw_header(player_done_random, main_surf)

#             print("player_1_wins: ", environment.player_1_won, " player_2_wins: ", environment.player_2_won)
#             print("games: ", games)
#         print("===>> player_1_wins: ", environment.player_1_won, " player_2_wins: ", environment.player_2_won)
#         return environment.player_1_won, environment.player_2_won        



#     def __call__(self, games_num):
#         return self.test(games_num)


# if __name__ == '__main__':
#     env = Environment()
#     player1 = DQN_Agent()
#     player2 = Random_Agent()
#     test = Tester(env,player1, player2)
#     print(test.test(100))

