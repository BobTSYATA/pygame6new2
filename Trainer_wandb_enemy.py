import pygame
import torch
from CONSTANTS import *
from Environment import Environment
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
# from Human_Agent import Human_Agent
# from Baseline_Agent import Baseline_Agent
# from TroopUnit import TroopUnit
from DQN import DQN
# import os
import wandb
# from State_ONLY_FOR_CALLING import State_ONLY_FOR_CALLING
from Tester import Tester
# from Tester_For_Enemy import Tester_For_Enemy

def log_metrics(epoch, avgloss, environment, player_1_wins, player_2_wins,differential):
    wandb.log({
        "avgloss": avgloss,
        "Wins/Player 1": player_1_wins,
        "Wins/Player 2": player_2_wins,
        "differential": differential,
        "Epoch": epoch
    })
    print(f"Metrics logged at epoch {epoch}.")

def save_weights_to_file(model, epoch):
    filename = f"weights_epoch_{epoch}.txt"
    with open(filename, "w") as f:
        for name, param in model.named_parameters():
            f.write(f"Layer: {name}\n")
            f.write(f"Values:\n{param.data.numpy()}\n")  # Weights/Biases
            f.write(f"Gradients:\n{param.grad.numpy() if param.grad is not None else 'None'}\n\n")  # Gradients

def save_episode_data(epoch, player_1_actions, player_2_actions, player_1_rewards, player_2_rewards, states, next_states, next_next_states, filename="episode_data.txt"):
    with open(filename, "a") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Player 1 Actions: {player_1_actions}\n")
        f.write(f"Player 2 Actions: {player_2_actions}\n")
        f.write(f"Player 1 Rewards: {player_1_rewards}\n")
        f.write(f"Player 2 Rewards: {player_2_rewards}\n")
        f.write(f"States: {states}\n")
        f.write(f"Next States: {next_states}\n")
        f.write(f"Next-Next States: {next_next_states}\n")
        f.write("=" * 50 + "\n")

# had to add reward for player2. restarting the run and at epoch 2k to check how things are going / to check if it gets stuck at an epoch. + when not working to check get next state and other things, etc...
RUN_NUM = 126 # run_num 125 won 74 lost 26 from 100 games. it started learning and learned only for 7k epochs, to run 100k and it's supposed to be perfect like player_1. # from run num 108 for player2 # 107 run_num == testing after changes #102 best run for 100k epochs
# at run num 113 added to next_state player num "2"/"1" accordingly. idk how it worked until now for Trainer_wandb. if not working to go over AGAIN the structure for THIS script, the Trainer_wandb_enemy.py and then go over functions at Environment or smt idk.
# at run num 114 added Tester_For_Enemy.py
# might have an issue with logging player 1 and player 2 wins at run num 115 they look almost exactly the same. to keep track.
# 1.to end the run not learning and for run 118 to do the .reverse like in Reversi for the state_1_R like in Trainer_black.py but to understand it first so to download it localy and put a breakpont one line 
# before the while loop starts and compare to my code too + put breakpoints before and after .reverse was used  
# + 2. A HUGE BUG WAS FOUND, THE RED ENEMY CAN SEND TO ANYTHING AND IT LEARNS TO ONLY SEND TO ITSELF FOR SOME REASON AND THE GAME NEVER STOPS
# + 2. continue: the red sends most of the time to island number 3, not to itself, always to 3, this is what it learns / some bug makes it do so. to fix it. (sometimes it sends to itself too with this action: (0,0) most of the time it's (3,3) because once conquered sending to itself)

# *. 2. to not use Tester_For_Enemy.py. supposed to use the normal Tester. --> [done]
# *. 3. only thing that i changed without testing is the tester: to this: Tester(player1=player, player2=Random_Agent(), env=environment,main_surf=main_surf) from this: Tester(player1=Random_Agent(), player2=player2, env=environment,main_surf=main_surf) --> [done]
# *. so to do the same here but the opposite ofc instead of tester for enemy --> [done]

########################################################################################RIGHT NOW TO DO:########################################################################################################
# run num 121 == runnign trainer_wandb.py for checking if it still learns well with new tester and player num in get next state, if NOT, to change back and check, and THEN to get the changes here, if learning with the changes first try to go try do the reverse thing + structure for this script etc...
# 1. player_num="2" needs to be everywhere / ="1" depending on the player but it needs to be consistent, in this script needs to be "2" and in Trainer_wandb needs to be "1". did this and running, 
# if working to rerun this script cause i changed player_num, if not working there to change tester back,
# and then rerun it and it's supposed to work and change the tester here accordingly + and rerun, and if still not working to do the reverse thing + structure for this script, etc... 
# doesn't seem to be learning player 1 after 3k epochs, to wait until 15k and see if something changed, if not then to change according to the instructions above
# okay so player 1 doesn't learn well, to redo the tester thingy, and THEN rerun, and if working THEN to rerun this one, and if not to do tester like in trainer_wandb.py here
# if still not learning for player1 to redo the move done thingy at the end. + and to get it to work and write everything changed and then rerun the player2 with correct changes and stuff.
# run_num 123 redid the done in def move in environment, if still not working SOMETHING CHANGED, might be the Tester script for get action etc.. might be [not Trainer_wandb, maybe Tester/Environment/smt else idk what probably the first two]
# run_num 124 i changed player_num = 1 from 2 at Tester to be the original num, to understand what get_next_state player_num needs to be if it needs to be changed to 2 or can stay the same. + wins for both players are working correctly of being lower then epoch because of tester usage.

# RIGHT NOW (for RUN_NUM = 125): 
# player1 learned. added the new done at def move in environment.  lines 84 + 56 at Tester i changed player_num from 1 to 2 because running it for this player2 trainer. + line 754 at Environment at move function changed to player num 2 same reason. 
# other changes are in THIS script structure but that's it: switched player1 and player2 agent types, added 1 action and stuff before while loop starts, switched all player_nums accordingly and all of the things for player1 learning now for player2.
# if not learning to put breakpoint before while and see if starting stage is the same one as in reversi, and to figure out how to make player2 learn.

# player 2 doesn't seem to learn to try with player num 1 at tester and next state/ see researchn and follow the black trainer at reversi and redo mine. especially reverse probably

################################################################################################################################################################################################################

path_load= None
path_Save=f'DataTraining/params_{RUN_NUM}.pth'
path_best = f'DataTraining/best_params_{RUN_NUM}.pth'
buffer_path = f'DataTraining/buffer_{RUN_NUM}.pth'
results_path=f'DataTraining/results_{RUN_NUM}.pth'
random_results_path = f'DataTraining/random_results_{RUN_NUM}.pth'
path_best_random = f'DataTraining/best_random_params_{RUN_NUM}.pth'

# runnign this script... not supposed to work ;--;. when not working to check get_next_state and other things, etc..
# switched player 1 and player 2 num in move function in environment no wonder the graphs were the same. to wait for 50k and see if it learned and then rerun it and rethink about the player_won at move function again to check i am correct. 
def main():
    pygame.init()
    environment = Environment()
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    main_surf.fill(LIGHTGRAY) 

    player2 = DQN_Agent(train=True, player_num=2)  # DQN agent for Player 2
    Q = player2.DQN
    Q_hat :DQN = Q.copy()
    Q_hat.train = False


    # Initialize Player 2 with a specific agent type (Human, Random, DQN)
    player = Random_Agent()
    # player2 = Baseline_Agent()

    batch_size = 64
    buffer = ReplayBuffer()
    learning_rate = 0.002 #0.001#0.00001 #0.001#0.0001#0.01#0.001#0.00001
    ephocs = 50000#1000#100000#50000#100000#1000#100#200000
    start_epoch = 1#0
    C = 100#200 #9#5#3
    avgLoss = 0
    loss = torch.Tensor([0])
    loss_count = 0
    tester = Tester(player1=Random_Agent(), player2=player, env=environment,main_surf=main_surf) #rnd vs rnd
    tester_fix = Tester(player1=player2, player2=player, env=environment,main_surf=main_surf)# tpye vs type
    random_results = []
    results = []
    best_random = -100
    res = 0
    best_res = -200
    epsiln_decay = 2000 # same one as in DQN_Agent
    avglosses = []
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)#, weight_decay=0.0005)
    step = 0



    checkpoint_path = f"DataTraining/checkpoint{RUN_NUM}.pth" 

    wandb.init(
        project="Island_Conquerer",
        resume=False, 
        id=f'Island_Conquerer {RUN_NUM}',
        config={
        "name": f"Island_Conquerer {RUN_NUM}",
        "checkpoint": checkpoint_path,
        "learning_rate": learning_rate,
        "architecture": "FNN 18, 64, 32, 1",
        "epochs": ephocs,
        "start_epoch": start_epoch,
        "decay": epsiln_decay,
        "gamma": 0.75,
        "batch_size": batch_size, 
        "C": C
        }
    )



    for epoch in range(start_epoch, ephocs):
        environment.restart()

        after_state_2 = None
        state = environment.set_init_state(player_num="1") # needs to be one

        print(f"Epoch {epoch}/{ephocs} starting...")
        done = False


        # Initialize lists to collect data for the entire epoch
        player_1_actions = []
        player_2_actions = []
        player_1_rewards = []
        player_2_rewards = []
        states_graphics = []
        next_states_graphics = []
        next_next_states_graphics = []

        # Player 1 takes an initial move before the loop
        action_tuple_1 = player.get_Action(environment, "1")
        r, d = environment.move(epoch,main_surf, action_tuple_1, agent_type="Random_Agent", player_num="1")

        # Record data for Player 1
        player_1_actions.append(action_tuple_1)
        player_1_rewards.append(r)
        states_graphics.append(state.Graphics)

        state = environment.get_next_state(player_num="2")

        while not done:
            # print(f"Step: {step}", end='\r')
            step += 1

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return

            environment.draw_header(done, main_surf)
            # Player 1's turn
            action_tuple_2 = player2.get_Action(environment, state=state, train=True, epoch=epoch)# events=events
            reward2, done = environment.move(epoch,main_surf, action_tuple_2, agent_type="DQN", player_num="2")
            # print(f"Epoch {epoch} | Player 2 Reward: {reward2}, Player 2 Done: {done}") # current reward
            # Record data for Player 2
            player_2_actions.append(action_tuple_2)
            player_2_rewards.append(reward2)
            next_states_graphics.append(state.Graphics)

            after_state = environment.get_next_state(player_num="2")
            
            if done:
                buffer.push(state, action_tuple_2, reward2, after_state, done) 
                break

            # Player 2's turn
            environment.draw_header(done, main_surf)

            action_tuple_1 = player.get_Action(environment, "1")#events
            reward1, done = environment.move(epoch,main_surf, action_tuple_1, agent_type="Random_Agent", player_num="1")

            # Record data for Player 1
            player_1_actions.append(action_tuple_1)
            player_1_rewards.append(reward1)
            next_next_states_graphics.append(after_state.Graphics)

            after_state_2 = environment.get_next_state(player_num="2")
            reward = reward1 + reward2 # needs to be used for the second buffer.push

            # if done:
            #     print("done_2: ", done)


            buffer.push(state, action_tuple_2, reward, after_state_2, done)

            state = after_state_2 # might cause problems 



            if epoch < batch_size:
                continue
            
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            Q_values = Q(states[0], actions)

            next_actions = player2.get_actions(environment, next_states, dones)
            with torch.no_grad():
                Q_hat_Values = Q(next_states[0], next_actions)

            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)

            optim.zero_grad() 
            loss.backward()
            optim.step()



            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001
                

        if (epoch+1) % 500 == 0:  # Check every 500 epochs
            print(f"Epoch {epoch} | Q-value range: min={Q_values.min().item()}, max={Q_values.max().item()}")

        if epoch % 458 == 0:
            save_episode_data(epoch, player_1_actions, player_2_actions, player_1_rewards, player_2_rewards, states_graphics, next_states_graphics,next_next_states_graphics)


        if epoch % 972 == 0:
            save_weights_to_file(Q,epoch)
            test = tester(100)
            test_score = test[0]-test[1]
            if best_random < test_score and tester_fix(1) == (0,1):
                best_random = test_score
                player2.save_param(path_best_random)
            print(test)
            random_results.append(test_score)


        if epoch % C == 0:
            print(f"Updating Q_hat at epoch {epoch}")
            Q_hat.load_state_dict(Q.state_dict())

        step = 0

        if (epoch+1) % 100 == 0: 
            avglosses.append(avgLoss)

            results.append(res)

            # print("===>> player_1_wins: ", environment.player_1_won, " player_2_wins: ", environment.player_2_won)
            differential = environment.player_2_won - environment.player_1_won
            log_metrics(epoch, avgLoss, environment, environment.player_1_won, environment.player_2_won,differential)

            if best_res < res:      
                best_res = res
                if best_res > 75 and tester_fix(1) == (0,1):
                    player2.save_param(path_best)
            res = 0

        if (epoch+1) % 4986 == 0:
            torch.save({'epoch': epoch, 'results': results, 'avglosses':avglosses}, results_path)
            # torch.save(buffer, buffer_path)
            player2.save_param(path_Save)
            torch.save(random_results, random_results_path)


    player2.save_param(checkpoint_path)

    torch.save({'epoch': epoch, 'results': results, 'avglosses':avglosses}, results_path)
    torch.save(random_results, random_results_path)


if __name__ == "__main__":
    main()
