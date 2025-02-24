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
from DQN import DQN
import os
import wandb
from State_ONLY_FOR_CALLING import State_ONLY_FOR_CALLING



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

def save_episode_data(epoch, player_1_actions, player_2_actions, player_1_rewards, player_2_rewards, states, next_states, filename="episode_data.txt"):
    with open(filename, "a") as f:
        f.write(f"Epoch: {epoch}\n")
        f.write(f"Player 1 Actions: {player_1_actions}\n")
        f.write(f"Player 2 Actions: {player_2_actions}\n")
        f.write(f"Player 1 Rewards: {player_1_rewards}\n")
        f.write(f"Player 2 Rewards: {player_2_rewards}\n")
        f.write(f"States: {states}\n")
        f.write(f"Next States: {next_states}\n")
        f.write("=" * 50 + "\n")

def main():
    pygame.init()
    environment = Environment()
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    main_surf.fill(LIGHTGRAY) #################################

    player = DQN_Agent(train=True, player_num=1)  # DQN agent for Player 1

    Q = player.DQN
    Q_hat :DQN = Q.copy()
    Q_hat.train = False


    # Initialize Player 2 with a specific agent type (Human, Random, DQN)
    player2 = Random_Agent()
    # player2 = Baseline_Agent()

    batch_size = 64#32#64#128#64#50 # idk why 64 it doesn't need to be the same as the DQN
    buffer = ReplayBuffer()
    learning_rate = 0.002 #0.001#0.00001 #0.001#0.0001#0.01#0.001#0.00001
    ephocs = 100000#50000#100000#1000#100#200000
    start_epoch = 1#0
    C = 100#200 #9#5#3
    avgLoss = 0
    loss = torch.Tensor([0])
    loss_count = 0
    #scores, losses, avg_score = [], [], []

    epsiln_decay = 2000 # same one as in DQN_Agent

    avglosses = []
    

    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)#, weight_decay=0.0005)



    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [2000, 3500, 5500, 7500, 9000], gamma=0.99) #[3000, 5000, 7000, 9000] #[1000, 2000, 3000, 5000, 7000, 9000], gamma=0.85 # updates the learning rate at: 100, 500, 1000, 2000, 3000, 5000, 7000, 9000, 
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optim, 
    #     max_lr=0.005,  # Peak LR (higher than base to start strong)
    #     total_steps=ephocs,  # Total training steps
    #     pct_start=0.1,  # First 10% of training will increase LR
    #     anneal_strategy='cos',  # Cosine decay for smooth slowing
    #     final_div_factor=10  # Ends at 0.0005 (1/10 of max LR)
    # )

    step = 0

    RUN_NUM = 102 # from 100 to 102 changed gamma from 0.6 to 0.75, cahnged epsiln_decay to 5000 from 2000 and epsilon_final to 0.01 from 0.05. also, running 100k epochs instead of 30k.
    ######### checkpoint Load ############
    checkpoint_path = f"DataTraining/checkpoint{RUN_NUM}.pth" 



    # checkpoint21: 128, 258, 512, 128, 64 death-=3 gamma 0.99 LR = 0.00001 Schedule: 5000, 10000, 15000 death -3 c = 3 decay = 20000
    wandb.init(
        # set the wandb project where this run will be logged
        project="Island_Conquerer",
        resume=False, 
        id=f'Island_Conquerer {RUN_NUM}',
        # track hyperparameters and run metadata
        config={
        "name": f"Island_Conquerer {RUN_NUM}",
        "checkpoint": checkpoint_path,
        "learning_rate": learning_rate,
        "architecture": "FNN 136, 322, 468, 64",#128, 258, 512, 128, 64, 4
        "Schedule": "2000, 3500, 5500, 7500, 9000 gamma=0.75", #1000, 2000, 3000, 5000, 7000, 9000
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
        # print("calling set_init_state from trainer wandb")
        state = environment.set_init_state(player_num="1")

        
        # Start of epoch
        print(f"Epoch {epoch}/{ephocs} starting...")
        done = False


        # Initialize lists to collect data for the entire epoch
        player_1_actions = []
        player_2_actions = []
        player_1_rewards = []
        player_2_rewards = []
        states_graphics = []
        next_states_graphics = []


        while not done:
            # print(f"Step: {step}", end='\r')
            step += 1

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return

            # reward1, done, after_state = environment.move(main_surf, action_tuple_1, agent_type="DQN", player_num="1")# TO CHANGE TO BE TURN BASED AND MAKE TROOPUNITS TO BE FAST IN CONSTANTS[second part in constants done]   #, state=state) # maybe the done_1 is wrong to debug later ask someone and think about it?
            environment.draw_header(done, main_surf)
            # Player 1's turn
            action_tuple_1 = player.get_Action(environment, state=state, events=events, train=True, epoch=epoch)
            reward1, done = environment.move(epoch,main_surf, action_tuple_1, agent_type="DQN", player_num="1")

            # Record data for Player 1
            player_1_actions.append(action_tuple_1)
            player_1_rewards.append(reward1)
            states_graphics.append(state.Graphics)


            after_state = environment.get_next_state()
            
            if done:
                # print("done_1: ",done)
                buffer.push(state, action_tuple_1, reward1, after_state, done) # to get after_state at environment.move if not working?
                break

            # Player 2's turn (only after Player 1’s action is resolved)
            environment.draw_header(done, main_surf)
            action_tuple_2 = player2.get_Action(environment, "2", events)
            reward2, done = environment.move(epoch,main_surf, action_tuple_2, agent_type="Random_Agent", player_num="2")

            # Record data for Player 2
            player_2_actions.append(action_tuple_2)
            player_2_rewards.append(reward2)
            next_states_graphics.append(after_state.Graphics)

            after_state_2 = environment.get_next_state()
            reward = reward1 + reward2 # needs to be used for the second buffer.push

            # if done:
            #     print("done_2: ", done)


            buffer.push(state, action_tuple_1, reward, after_state_2, done)

            state = after_state_2 # might cause problems 



            if epoch < batch_size:
                continue
            
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)
            # print(f"Sampled batch - States: {states[0][:5].detach().cpu().numpy()}, Actions: {actions[:5].detach().cpu().numpy()}, Rewards: {rewards[:5].detach().cpu().numpy()}")

            Q_values = Q(states[0], actions)
            # print("Q_values: ", Q_values)
            # print(f"11111111111111111111111111111111111 Epoch {epoch} | Q_values sample: {Q_values[:5].detach().cpu().numpy()}")  # Print first 5

            next_actions = player.get_actions(environment, next_states, dones)# next_actions most are similar?
            # print("next_actions: ", next_actions)
            with torch.no_grad():
                Q_hat_Values = Q(next_states[0], next_actions)#Q_hat(next_states[0], next_actions)
                # print(f"Epoch {epoch} | Target Q-values (Q_hat_Values) sample: {Q_hat_Values[:5].detach().cpu().numpy()}")


            # print("Q_hat_Values: ", Q_hat_Values) # needs to be a list with 1 value in each place i think
            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)
            # print("loss: ", loss)
            optim.zero_grad() # trying with zero_grad infront but it's supposed to be after optim.step() + trying it to attack island num 1 alwaysfor run_num 75 if not working to check basic environmet stuff + deep debugging and printing
            # print("1 loss: ", loss)

            # # Gradient Clipping to prevent extreme updates
            # torch.nn.utils.clip_grad_norm_(Q.parameters(), max_norm=1.0)

            loss.backward()
            # print("2 loss: ", loss)
            # before_update = sum(p.sum().item() for p in Q.parameters())
            optim.step()
            # after_update = sum(p.sum().item() for p in Q.parameters())
            # print(f"Weight Change: {after_update - before_update}")
            # print("optim.step() 2: ", optim.step())
            # optim.zero_grad() # it does need to be here i think
            # scheduler.step()

            



            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001
                

        if (epoch+1) % 500 == 0:  # Check every 500 epochs
            print(f"Epoch {epoch} | Q-value range: min={Q_values.min().item()}, max={Q_values.max().item()}")

        if epoch % 458 == 0:# 458 random so won't give the error
            save_episode_data(epoch, player_1_actions, player_2_actions, player_1_rewards, player_2_rewards, states_graphics, next_states_graphics)


        if epoch % 972 == 0: # 1972 random so won't give the error
            save_weights_to_file(Q,epoch)

            # save_episode_data(epoch, player_1_actions, player_2_actions, player_1_rewards, player_2_rewards, states_graphics, next_states_graphics)


        if epoch % C == 0:
            print(f"Updating Q_hat at epoch {epoch}")
            Q_hat.load_state_dict(Q.state_dict())


        # print(f'Epoch: {epoch} | Loss: {loss.item():.7f} | LR: {scheduler.get_last_lr()} | Step: {step} | wins1: {environment.player_1_won}')#  | Score: {environment.score} | Best Score: {best_score} 
        # print(f'Epoch: {epoch} | Loss: {loss.item():.7f} | Step: {step} | wins1: {environment.player_1_won}')
        #print(f"\033[done_1: {done_1}.\033[0m")


        step = 0

        if (epoch+1) % 100 == 0: 
            avglosses.append(avgLoss)
            # print("===>> player_1_wins: ", environment.player_1_won, " player_2_wins: ", environment.player_2_won)
            differential = environment.player_1_won - environment.player_2_won
            log_metrics(epoch, avgLoss, environment, environment.player_1_won, environment.player_2_won,differential)


    player.save_param(checkpoint_path)

if __name__ == "__main__":
    main()

            # move function line 653 + 660 + 661 to change to this reward if not working and try again, and if still not working to do the line under this line, etc..
            # to compare to checkers everything one by one from rewards to dqn to replaybuffer to state to dqnagent to trainer, etc. 
            # then to duplicate it almost in my head and try to check why my things are different(leaky relu for example because of zeros, to check.) and i need a working trained agent for player 1 that looks and works perfectly and then to replicate to player 2 and then the tik proyect, etc..
            ######################################################################################################################################################
            # i was able to teach the agent to attack island num 1 all of the time, now with the new reward system that prioritizes attacking the island with least troops with a few other conditions will be checked --> [trying now]
            # before i check if it works/doesn't work, ill need to:
            # 1.check everything with relu instead of leaky_relu. --> [results: most gradients were 0, like 95% of gradient values were zeros or 100% were zeros, this is why i switched to leaky relu, maybe with new reward system CAN go back to relu but will not try rn.]
            # 2.to put zero_grad afterwards --> [result: made all gradients to be None, idk how it works for space invaders, etc, to check later on after finishing the checks.]
            # 3.to learn about linear bias in the weights file print. --> []
            # 4.to check if it successfully saves the params with no erros. --> []
            # 5.to check line 32 in this code and see if it's working. --> didn't work so deleted, it was supposed to get rid of the ... when python prints large strings, idc about it though. ;]
            # 6.to check the avgloss and understand why it doesn't look correct. --> []
            ######################################################################################################################################################






            # every 1000 epochs to save the gradient and the weights like i did + every 333 epochs to save all of the actions for both players for the whole game (everything, maybe from the replaybuffer: reward, done, action+ x in forward dqn, state)

            # RunNum75.first run of 100k epochs:https://wandb.ai/tsyata1-none/Island_Conquerer/runs/Island_Conquerer%2075?nw=nwusertsyata1 (testing and trying to see if it will attack island number 1 all of the time)
            # results: i have seen that at the avgloss graph e starts to go down many times, but every time it does it starts to go up again. this is probably because of the high learning rate / batch size. 
            # RunNum76.second run for 50k epochs:https://wandb.ai/tsyata1-none/Island_Conquerer/runs/Island_Conquerer%2076?nw=nwusertsyata1 (testing with lower learning rate and 64 instead of 32 batch size)
            # results: stopped at epoch 2398, not really anything important. chaging reward to += 5 and -=100
            # RunNum77. third for run 50k epochs:https://wandb.ai/tsyata1-none/Island_Conquerer/runs/Island_Conquerer%2077?nw=nwusertsyata1 -->
            # --> (testing with huge negative reward and small positive reward ONLY WANTING TO MAKE IT SEND TO ISLAND NUM 1, if not working try +=5 and +=0.05 for other islands and that is not working to do line 222) # to big of a reward can cause issues to try +=2 and -=6
            # results:i found out that i have been using DDQN all of this time trying to convert back to DQN (not q hat copy)
            # RunNum78: trying DQN structure: no DDQN, no Q_Hat updating, etc. maybe to go back soon to DDQN if won't work, and to do the prints and all.. running for 50k epochs:
            # to try next with +=0.05 and +=5 instead of negative reward -=6
            # RunNum82: running 50k epochs:  
            # changed the loss function added .max(1)[0].detach() because the gradients were None/Zeros all the time and it fixed it.
            # results: still some 0's in the gradient
            # RunNum84: using leakyRelu at forward in dqn because:When the data has a lot of noise or outliers: Leaky ReLU can provide a non-zero output for negative input values, which can help to avoid discarding potentially important information, and thus perform better than ReLU in scenarios where the data has a lot of noise or outliers.
            # results:
            # RunNum87: i have found out that the Q_target(q_new) values were not 1, but a bunch of numbers, so i took the .max one, also changed reward system values, and i have a list of 6 things i need to check:
            # BUT BEFORE THAT TO CHECK THE Q_VALUES AT DQN AND DQN AGENT the avgloss still goes up...AND IDK TO CHECK EVERYTHING, ALL THE VALUES EVERYWHERE xd
            ##########################################################################################################################################
            # 5️⃣ Check Replay Buffer and Sampled Data
            # 🔍 What to check?
            # If the replay buffer stores incorrect data, your agent will never learn.
            # 📍 Where to add print statements?
            # Inside ReplayBuffer.py, inside push:

            # print(f"Stored transition: State {state.Graphics[:5]}, Action {action}, Reward {reward}, Next State {next_state.Graphics[:5]}, Done {done}")
            # Inside Trainer_wandb.py, before sampling from replay buffer:

            # print(f"Sampled batch - States: {states[0][:5].detach().cpu().numpy()}, Actions: {actions[:5].detach().cpu().numpy()}, Rewards: {rewards[:5].detach().cpu().numpy()}")
            # ✅ If the buffer is filled with the same states/actions, the agent isn’t exploring.
            # ✅ If rewards don’t make sense, check the reward function.
            ##########################################################################################################################################



            # to do: think about the reward system, maybe give him a bigger reward like 10 if the amount of troops he sent to an enemy island is bigger then the amount of troops on the island AND the amount
            # of troops that are the enemys that the island is their destination. and more stuff like that to think about. my game i can change however i want.
            # TO DO: to check everything, to check the after states and to see if i get them correctly and stuff, and to check for bad structure or things i forgot to add/do.
            # reward system: first thing to do add on top of the already given reward this:
            # if sent_troops > (defending_troops + incoming_enemy_troops):
            #     reward += c * (sent_troops - (defending_troops + incoming_enemy_troops))
            # second thing in the reward system is this:
            # if reinforcing_TroopUnit and not is_under_threat(reinforcing_TroopUnit):
            #     reward -= small_penalty
            # def is_under_threat(troopUnit):
            #     for enemy_troopUnit in enemy_troopUnits:
            #         if enemy_troopUnit.destination == troopUnit.destination:
            #             return True
            #     return False
            # AND SAME THING FOR ATTACK, IF ALREADY SENDING TROOP UNITS FOR AN ISLAND THAT THE FIRST THING I WROTE IS TRUE:if sent_troops > (defending_troops + incoming_enemy_troops):
            # TO GIVE A NEGATIVE REWARD BECAUSE IT IS UNECESSARY TO SEND EVEN MORE TROOP UNITS TO AN ALREADY GURANTEED CONQUER




        # https://github.com/Ben124125/Checkers_DQN/blob/main/DQN_Agent.py
        # https://github.com/MarkmanGilad/Reversi_AI_new/blob/main/Trainer_black.py
        # https://github.com/MarkmanGilad/Space_Invaders_wandb/blob/main/Trainer_wandb.py


