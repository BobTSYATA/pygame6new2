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


def log_metrics(epoch, avgloss, environment, player_1_wins, player_2_wins):
    wandb.log({
        "avgloss": avgloss,
        "Wins/Player 1": player_1_wins,
        "Wins/Player 2": player_2_wins,
        "Epoch": epoch
    })
    print(f"Metrics logged at epoch {epoch}.")

def main():
    pygame.init()
    environment = Environment()
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    main_surf.fill(LIGHTGRAY) #################################

    #best_score = 0

    ### to do: save everything and redo this part from Trainer_black.py in reversi. ##
    ####### params ############
    player = DQN_Agent()  # DQN agent for Player 1

    ###### CHANGING STRUCTURE ######
    # player_hat = DQN_Agent()
    # player_hat.DQN = player.DQN.copy()
    ###### CHANGING STRUCTURE ######
    Q = player.DQN
    Q_hat :DQN = Q.copy()
    Q_hat.train = False


    # Initialize Player 2 with a specific agent type (Human, Random, DQN)
    player2 = Random_Agent()
    # player2 = Baseline_Agent()

    batch_size = 32#64#128#64#50 # idk why 64 it doesn't need to be the same as the DQN
    buffer = ReplayBuffer()
    learning_rate = 0.01#0.001#0.00001
    ephocs = 10000#1000#100#200000
    start_epoch = 0
    C = 200 #9#5#3
    avgLoss = 0
    loss = torch.Tensor([0])
    loss_count = 0
    #scores, losses, avg_score = [], [], []



    player_1_wins, player_2_wins, avglosses = [], [], []
    
    ###### CHANGING STRUCTURE ######
    #optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
    ###### CHANGING STRUCTURE ######
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)



    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [3000, 5000, 7000, 9000], gamma=0.99) #[3000, 5000, 7000, 9000] #[1000, 2000, 3000, 5000, 7000, 9000], gamma=0.85 # updates the learning rate at: 100, 500, 1000, 2000, 3000, 5000, 7000, 9000, 
    step = 0

    RUN_NUM = 56
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
        "Schedule": "3000, 5000, 7000, 9000 gamma=0.99", #1000, 2000, 3000, 5000, 7000, 9000
        "epochs": ephocs,
        "start_epoch": start_epoch,
        "decay": epsiln_decay,
        "gamma": 0.99,
        "batch_size": batch_size, 
        "C": C
        }
    )
    
    #################################

    print(f"Starting training from epoch {start_epoch} to {ephocs}")

    #################################

    for epoch in range(start_epoch, ephocs):
        environment.restart()
        end_of_game = False

        state = environment.set_init_state()

        
        # Start of epoch
        print(f"Epoch {epoch}/{ephocs} starting...")

        while not end_of_game:

            print(f"Step: {step}", end='\r')
            step += 1

            # state = environment.state()
            # print("1 state.player: ",state.player)
            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return

            ############## Sample Environment #########################
            # Player 1's move (DQN)
            action_tuple_1 = player.get_Action(environment, state=state, events=events, train=True)#player.get_Action(environment, events=events, state=state)
            #print("action_tuple_1: ",action_tuple_1)

            ####new####
            reward1, done_1 = environment.move(action_tuple_1, agent_type="DQN", player_num="1", state=state) # maybe the done_1 is wrong to debug later ask someone and think about it?
            #print("reward1: ", reward1)

            environment.draw_header(done_1, main_surf) #update the graphics before getting the next state?
            environment.update_troop_units()

            #conquering_troops = environment.update_troop_units()
          
            #after_state = environment.state()
            after_state = environment.next_state(state)#set_init_state() # because it is a half action game #environment.next_state(state, action_tuple_1) # to call init state again instead of next_state?
            # print("2 state.player: ",state.player)
            if done_1:
                buffer.push(state, action_tuple_1, reward1, after_state, done_1) # to get after_state at environment.move if not working?
                break

            #environment.draw_header(done_1, main_surf) #unneeded? to remove for now 
            
          
            action_tuple_2 = player2.get_Action(environment, "2", events)
            # print("3 state.player: ",state.player)
            # print("action_tuple_2: ", action_tuple_2, " action_tuple_2[-1]: ", action_tuple_2[-1])
            reward2, done_2 = environment.move(action_tuple_2, agent_type="Random_Agent", player_num="2",state=state)
            #print("reward2: ", reward2)
            environment.draw_header(done_2, main_surf)
            environment.update_troop_units()
            
            after_state_2 = environment.next_state(after_state)#set_init_state() # because it is a half action game #environment.next_state(state, action_tuple_1) #environment.state()
            # print("4 state.player: ",state.player)
            reward = reward1 + reward2 # needs to be used for the second buffer.push

            if done_2:
                #TroopUnit.TROOPUNIT_NUM = 0
                break  # End the game if any player is done

            buffer.push(state, action_tuple_2, reward, after_state_2, done_2)

            state = after_state_2 # might cause problems 

            environment.draw_header(done_2, main_surf) # unneeded i think but whatever #is this the state = next_state ?
            environment.update_troop_units() # unneeded i think but whatever


            # if len(buffer) < MIN_BUFFER:
            #     #print(f"Buffer size too small: {len(buffer)}")
            #     continue
            if epoch < batch_size:
                continue

            ### to do: save everything and redo this part from Trainer_black.py in reversi. ##
            states, actions, rewards, next_states, dones = buffer.sample(batch_size) # calls the replay buffer that causes the error # , dones


            # until here good to try to do it like Checkers: change get_Actions so it will actually also return the next_actions and with it create q hat values etc etc
            ###### CHANGING STRUCTURE ######
            # Q_values = player.Q(states, actions)
            # next_actions, Q_hat_Values = player_hat.get_Actions_Values(next_states)
            #loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
            ###### CHANGING STRUCTURE ######
            # Q_values = Q(states, actions) # states[0] in my game state doesn't hold action to fix later and add this back? to add +2 to dqn and all possible actions for state and at ReplayBuffer to seperate them like at checkers code?
            # next_actions = player.get_actions(next_states, dones) # to create the get_actions function at dqn agent
            # with torch.no_grad():
            #     Q_hat_Values = Q_hat(next_states, next_actions) # next_states[0] in my game state doesn't hold action to fix later and add this back? to add +2 to dqn and all possible actions for state and at ReplayBuffer to seperate them like at checkers code?

            # to check player num changes --> [to check more thoroughly]
            # to check if at DQN_Agent the get_action actions gets the correct actions for the current state and if not to pass state[0] somehow --> TO DO
            # to fix the screen g'itter --> [half way, fixed it so it only jitters at the end, to completely fix this]
            # to clean most of the comments that are irrelevent
            # to check if the values are correct like the done, reward, etc etc
            # to run it while i'm at martial arts class and to fix when it won't learn probably :[ 

            Q_values = Q(states[0], actions)
            next_actions = player.get_actions(environment, next_states, dones)
            with torch.no_grad():
                Q_hat_Values = Q_hat(next_states[0], next_actions)


            loss = Q.loss(Q_values, rewards, Q_hat_Values, dones)

            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 

        # now going to do:
        # at the structure change to do this: to create the get actions at the DQN Agent, to add the state[0] and state[1] and to add to the state function at environment all of the legal actions for both players and to add + 2 at the DQN input size and change everything else accordingly so it will work like:
        # when in this code above when i get the state to get the state[0], etc etc..
        # and to check the replaybuffer after if it's still not working

        # right now trying to convert it to be half like checkers (kinda 2.) i surrounded each one with this text: ###### CHANGING STRUCTURE ######
        # go over the replaybuffer and print istype for every single one and convert the necessery ones to Tensor /np.array and THEN to: [ ] 3.
        # fix the sample of the ReplayBuffer when i know what each thing is and how it will look at it and THEN to: [ ] 4.
        # go over this fucntion: get_Actions_Values at DQN_Agent and check what it does and how to fix it based on the 3 given examples, mainly based on Space_Invaders and THEN to: [ ] 2.
        # print the rewards and check the END OF THE GAME rewards with the -100 and the +100 and check if they actually get it because at places like the dqn agent i return None, None for the action, so to make sure the rewards are good and THEN to: [ done ] --> completly didn't work but now it does lol
        # to check that the STATE that the state() is actually CORRECT, to check the thing with the MAX TROOPS, etc and what happens to the cancelled actions, basically to check the state is correct, and THEN to: [ done ] --> didn't reach 136 most of the time because the matches were so short so maybe in the future to recheck but not needed probably works
        # to check why the loss is huge sometimes and sometimes normal looking under a 100 and THEN to: [ ] 5. --> loss calculation is alright the problem is with the Q values which is from the problems above probably
        # when done all above:
        # try to give the reward like the space invaders and not like a turn based game, yes there is an enemy but MAYBE i should ignore its actions, idk, and THEN to: [ ]
        # idk to search online to try and figure this out after debugging and being pretty sure that this should work if it still doesn't. [ ]
        # https://github.com/Ben124125/Checkers_DQN/blob/main/DQN_Agent.py
        # https://github.com/MarkmanGilad/Reversi_AI_new/blob/main/Trainer_black.py
        # https://github.com/MarkmanGilad/Space_Invaders_wandb/blob/main/Trainer_wandb.py

        if epoch % C == 0:
            Q_hat.load_state_dict(Q.state_dict())

            ###### CHANGING STRUCTURE ######
            #player_hat.fix_update(dqn=player.DQN)
            ###### CHANGING STRUCTURE ######


            #player_hat.DQN.load_state_dict(player.DQN.state_dict())


        # Print stats after every epoch

        print(f'Epoch: {epoch} | Loss: {loss.item():.7f} | LR: {scheduler.get_last_lr()} | Step: {step} | wins1: {environment.player_1_won}')#  | Score: {environment.score} | Best Score: {best_score} 
        #print(f"\033[done_1: {done_1}.\033[0m")
        step = 0
        # Save stats every 10 epochs
        if (epoch+1) % 100 == 0: 
            avglosses.append(avgLoss)
            print("===>> player_1_wins: ", environment.player_1_won, " player_2_wins: ", environment.player_2_won)
            log_metrics(epoch, avgLoss, environment, environment.player_1_won, environment.player_2_won)


    player.save_param(checkpoint_path)

if __name__ == "__main__":
    main()





















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
# import os
# import wandb


# def save_checkpoint(epoch, player, optim, scheduler, buffer, avglosses, player_1_wins, player_2_wins, checkpoint_path, buffer_path):
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': player.DQN.state_dict(),
#         'optimizer_state_dict': optim.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'avglosses': avglosses,
#         'player_1_wins': player_1_wins,
#         'player_2_wins': player_2_wins,
#     }
#     torch.save(checkpoint, checkpoint_path)
#     torch.save(buffer, buffer_path)
#     print(f"Checkpoint saved at epoch {epoch}.")

# def log_metrics(epoch, avgloss, environment, player_1_wins, player_2_wins):
#     wandb.log({
#         "avgloss": avgloss,
#         "Wins/Player 1": player_1_wins,
#         "Wins/Player 2": player_2_wins,
#         "Epoch": epoch
#     })
#     print(f"Metrics logged at epoch {epoch}.")

# def main():
#     pygame.init()
#     environment = Environment()
#     main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
#     main_surf.fill(LIGHTGRAY) #################################

#     #best_score = 0

#     ### to do: save everything and redo this part from Trainer_black.py in reversi. ##
#     ####### params ############
#     player = DQN_Agent()  # DQN agent for Player 1
#     player_hat = DQN_Agent()
#     player_hat.DQN = player.DQN.copy()

#     # Initialize Player 2 with a specific agent type (Human, Random, DQN)
#     player2 = Random_Agent()
#     # player2 = Baseline_Agent()

#     batch_size = 128#64#50 # idk why 64 it doesn't need to be the same as the DQN
#     buffer = ReplayBuffer(path=None)
#     learning_rate = 0.1#0.001#0.00001
#     ephocs = 10000#1000#100#200000
#     start_epoch = 0
#     C = 9#5#3
#     avgLoss = 0
#     loss = torch.Tensor([0])
#     loss_count = 0
#     #scores, losses, avg_score = [], [], []



#     player_1_wins, player_2_wins, avglosses = [], [], []
    
#     optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [3000, 5000, 7000, 9000], gamma=0.85) #[1000, 2000, 3000, 5000, 7000, 9000], gamma=0.85 # updates the learning rate at: 100, 500, 1000, 2000, 3000, 5000, 7000, 9000, 
#     step = 0

#     RUN_NUM = 40

#     ######### checkpoint Load ############
#     checkpoint_path = f"DataTraining/checkpoint{RUN_NUM}.pth" 
#     buffer_path = f"DataTraining/buffer{RUN_NUM}.pth" 
#     if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
#         try:
#             checkpoint = torch.load(checkpoint_path)
#             start_epoch = checkpoint['epoch'] + 1
#             player.DQN.load_state_dict(checkpoint['model_state_dict'])
#             player_hat.DQN.load_state_dict(checkpoint['model_state_dict'])
#             optim.load_state_dict(checkpoint['optimizer_state_dict'])
#             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#             buffer = torch.load(buffer_path)
#             avglosses = checkpoint['avglosses']
#             player_1_wins = checkpoint['player_1_wins']
#             player_2_wins = checkpoint['player_2_wins']
#         except Exception as e:
#             print(f"Error loading checkpoint: {e}")
#             start_epoch = 0
#             buffer = ReplayBuffer(path=None)
#             avglosses, player_1_wins, player_2_wins = [], [], [] 
#     else:
#         print("No valid checkpoint found, starting fresh.")
#         start_epoch = 0
#         buffer = ReplayBuffer(path=None)
#         avglosses, player_1_wins, player_2_wins = [], [], []
#     player.DQN.train()
#     player_hat.DQN.eval()


#     # checkpoint21: 128, 258, 512, 128, 64 death-=3 gamma 0.99 LR = 0.00001 Schedule: 5000, 10000, 15000 death -3 c = 3 decay = 20000
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="Island_Conquerer",
#         resume=False, 
#         id=f'Island_Conquerer {RUN_NUM}',
#         # track hyperparameters and run metadata
#         config={
#         "name": f"Island_Conquerer {RUN_NUM}",
#         "checkpoint": checkpoint_path,
#         "learning_rate": learning_rate,
#         "architecture": "FNN 136, 322, 468, 64",#128, 258, 512, 128, 64, 4
#         "Schedule": "3000, 5000, 7000, 9000 gamma=0.85", #1000, 2000, 3000, 5000, 7000, 9000
#         "epochs": ephocs,
#         "start_epoch": start_epoch,
#         "decay": epsiln_decay,
#         "gamma": 0.85,
#         "batch_size": batch_size, 
#         "C": C
#         }
#     )
    
#     #################################

#     print(f"Starting training from epoch {start_epoch} to {ephocs}")

#     #################################

#     for epoch in range(start_epoch, ephocs):
#         environment.restart()
#         end_of_game = False
        

        
#         # Start of epoch
#         print(f"Epoch {epoch}/{ephocs} starting...")

#         while not end_of_game:

#             print(f"Step: {step}", end='\r')
#             step += 1

#             state = environment.state()

#             events = pygame.event.get()
#             for event in events:
#                 if event.type == pygame.QUIT:
#                     return

#             ############## Sample Environment #########################
#             # Player 1's move (DQN)
#             action_tuple_1 = player.get_Action(environment, player_num="1", events=events, state=state)

#             ####new####
#             done_1 = environment.move(action_tuple_1, agent_type="DQN", player_num="1", state=state) # maybe the done_1 is wrong to debug later ask someone and think about it?
            



#             conquering_troops = environment.update_troop_units()
#             #print(f"len(conquering_troops) = {len(conquering_troops)}, conquering_troops: {conquering_troops}")
#             #next_state = environment.state() # getting in environment and giving it to the TroopUnit in send_troops

#             # Updated implementation
#             # try:
#             #     #print(f"len(conquering_troops) = {len(conquering_troops)}, conquering_troops: {conquering_troops}")
#             for troop in conquering_troops:
#                 #print(f"Troop being added to buffer: ID={troop.id}")
#                 #print(f"State: {troop.created_state}, Action: {troop.created_action}, Reward: {troop.reward}, Next State: {troop.next_state}, Done: {troop.done}")
                
#                 buffer.push(
#                     troop.created_state,
#                     torch.tensor(troop.created_action, dtype=torch.int64),
#                     torch.tensor(troop.reward, dtype=torch.float32),
#                     troop.next_state,
#                     torch.tensor(troop.done, dtype=torch.float32)
#                 )
#                     #print("Buffer push successful.")
#             # except Exception as e:
#             #     print("Buffer push failed with error:", e)

#             # print(f"len(conquering_troops) = {len(conquering_troops)}")

#             # Old implementation
#             # for troop in conquering_troops:   
#             #     print("Troop being added to buffer:", troop)
#             #     print(f"ID: {troop.id}, State: {troop.created_state}, Reward: {troop.reward}, Action: {troop.created_action}, Next State: {troop.next_state}")     
#             #     buffer.push(troop.created_state, torch.tensor(troop.created_action, dtype=torch.int64), torch.tensor(troop.reward, dtype=torch.float32),
#             #         troop.next_state, torch.tensor(troop.done, dtype=torch.float32)) # , torch.tensor(done_1, dtype=torch.float32)
#             # print(f"len(Conquering troops) = {len(conquering_troops)}")
#             ####new####

#             #reward_1, done_1 = environment.move(action_tuple_1, agent_type="DQN", player_num="1", state=state)
#             if done_1:
#                 #TroopUnit.TROOPUNIT_NUM = 0
#                 break


#             environment.draw_header(done_1, main_surf) #################################
            
          
#             action_tuple_2 = player2.get_Action(environment, "2", events)

#             # print("action_tuple_2: ", action_tuple_2, " action_tuple_2[-1]: ", action_tuple_2[-1])
#             done_2 = environment.move(action_tuple_2, agent_type="Random_Agent", player_num="2",state=state)


#             ### to do: save everything and add buffer.push with negative reward values like in reversi.##
#             ### add the negative values from the update_troop_units function, and##
#             ### to add here the conquering_troops = environment.update_troop_units() like from above. ##


#             if done_2:
#                 #TroopUnit.TROOPUNIT_NUM = 0
#                 break  # End the game if any player is done

#             environment.draw_header(done_2, main_surf) #################################


#             if len(buffer) < MIN_BUFFER:
#                 #print(f"Buffer size too small: {len(buffer)}")
#                 continue
 

#             ### to do: save everything and redo this part from Trainer_black.py in reversi. ##
#             states, actions, rewards, next_states, dones = buffer.sample(batch_size) # , dones
#             Q_values = player.DQN(states)
#             Q_hat_Values = player_hat.DQN(next_states)
#             loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones) # , dones
#             loss.backward()
#             optim.step()
#             optim.zero_grad()
#             scheduler.step()

#             if loss_count <= 1000:
#                 avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
#                 loss_count += 1
#             else:
#                 avgLoss += (loss.item()-avgLoss)* 0.00001 






#         if epoch % C == 0:
#             # player_hat.fix_update(dqn=player.DQN)
#             player_hat.DQN.load_state_dict(player.DQN.state_dict())

#         # Print stats after every epoch

#         print(f'Epoch: {epoch} | Loss: {loss.item():.7f} | LR: {scheduler.get_last_lr()} | Step: {step} | wins1: {environment.player_1_won}')#  | Score: {environment.score} | Best Score: {best_score} 
#         #print(f"\033[done_1: {done_1}.\033[0m")
#         step = 0
#         # Save stats every 10 epochs
#         if (epoch+1) % 100 == 0: 
#             avglosses.append(avgLoss)
#             print("===>> player_1_wins: ", environment.player_1_won, " player_2_wins: ", environment.player_2_won)
#             log_metrics(epoch, avgLoss, environment, environment.player_1_won, environment.player_2_won)

            

#         # Save checkpoint every 1000 epochs
#         if epoch % 1000 == 0 and epoch > 0:
#             save_checkpoint(epoch, player, optim, scheduler, buffer, avglosses, environment.player_1_won, environment.player_2_won, checkpoint_path, buffer_path)

# if __name__ == "__main__":
#     main()










































# import pygame
# import torch
# from CONSTANTS import *
# from Environment import Environment
# from DQN_Agent import DQN_Agent
# from ReplayBuffer import ReplayBuffer
# from Random_Agent import Random_Agent
# from Human_Agent import Human_Agent
# import os
# import wandb

# def save_checkpoint(epoch, player, optim, scheduler, buffer, losses, player_1_wins, player_2_wins, checkpoint_path, buffer_path):
#     checkpoint = {
#         'epoch': epoch,
#         'model_state_dict': player.DQN.state_dict(),
#         'optimizer_state_dict': optim.state_dict(),
#         'scheduler_state_dict': scheduler.state_dict(),
#         'loss': losses,
#         'player_1_wins': player_1_wins,
#         'player_2_wins': player_2_wins,
#     }
#     torch.save(checkpoint, checkpoint_path)
#     torch.save(buffer, buffer_path)
#     print(f"Checkpoint saved at epoch {epoch}.")

# def log_metrics(epoch, loss, environment, player_1_wins, player_2_wins):
#     wandb.log({
#         "loss": loss.item(),
#         "Wins/Player 1": environment.player_1_won,
#         "Wins/Player 2": environment.player_2_won,
#         "Epoch": epoch
#     })
#     print(f"Metrics logged at epoch {epoch}.")

# def main():
#     pygame.init()
#     environment = Environment()
#     main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
#     main_surf.fill(LIGHTGRAY) #################################

#     best_score = 0

#     ####### params ############
#     player = DQN_Agent()  # DQN agent for Player 1
#     player_hat = DQN_Agent()
#     player_hat.DQN = player.DQN.copy()

#     # Initialize Player 2 with a specific agent type (Human, Random, DQN)
#     player2 = Random_Agent()

#     batch_size = 50
#     buffer = ReplayBuffer(path=None)
#     learning_rate = 0.00001
#     ephocs = 1000#100#200000
#     start_epoch = 0
#     C = 3
#     loss = torch.tensor(-1)
#     avg = 0
#     #scores, losses, avg_score = [], [], []
#     player_1_wins, player_2_wins, losses = [], [], []
    
#     optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [10000 * 1000], gamma=0.5)
#     step = 0

#     ######### checkpoint Load ############
#     checkpoint_path = "DataTraining/checkpoint4.pth" 
#     buffer_path = "DataTraining/buffer4.pth" 
#     if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
#         try:
#             checkpoint = torch.load(checkpoint_path)
#             start_epoch = checkpoint['epoch'] + 1
#             player.DQN.load_state_dict(checkpoint['model_state_dict'])
#             player_hat.DQN.load_state_dict(checkpoint['model_state_dict'])
#             optim.load_state_dict(checkpoint['optimizer_state_dict'])
#             scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
#             buffer = torch.load(buffer_path)
#             losses = checkpoint['loss']
#             player_1_wins = checkpoint['player_1_wins']
#             player_2_wins = checkpoint['player_2_wins']
#         except Exception as e:
#             print(f"Error loading checkpoint: {e}")
#             start_epoch = 0
#             buffer = ReplayBuffer(path=None)
#             losses, player_1_wins, player_2_wins = [], [], []
#     else:
#         print("No valid checkpoint found, starting fresh.")
#         start_epoch = 0
#         buffer = ReplayBuffer(path=None)
#         losses, player_1_wins, player_2_wins = [], [], []
#     player.DQN.train()
#     player_hat.DQN.eval()


    

#     ################# Wandb.init #####################
    
#     # checkpoint21: 128, 258, 512, 128, 64 death-=3 gamma 0.99 LR = 0.00001 Schedule: 5000, 10000, 15000 death -3 c = 3 decay = 20000
#     wandb.init(
#         # set the wandb project where this run will be logged
#         project="Island_Conquerer",
#         resume=False, 
#         id='Island_Conquerer 4',
#         # track hyperparameters and run metadata
#         config={
#         "name": "Island_Conquerer 4",
#         "checkpoint": checkpoint_path,
#         "learning_rate": learning_rate,
#         "architecture": "FNN 128, 258, 512, 128, 64, 4",
#         "Schedule": "5000, 10000, 15000 gamma=0.5",
#         "epochs": ephocs,
#         "start_epoch": start_epoch,
#         "decay": epsiln_decay,
#         "gamma": 0.99,
#         "batch_size": batch_size, 
#         "C": C
#         }
#     )
    
#     #################################

#     print(f"Starting training from epoch {start_epoch} to {ephocs}")

#     #################################

#     for epoch in range(start_epoch, ephocs):
#         environment.restart()
#         end_of_game = False
        
        
#         # Start of epoch
#         print(f"Epoch {epoch}/{ephocs} starting...")

#         while not end_of_game:
#             print(f"Step: {step}", end='\r')
#             step += 1

#             state = environment.state()

#             events = pygame.event.get()
#             for event in events:
#                 if event.type == pygame.QUIT:
#                     return

#             ############## Sample Environment #########################
#             # Player 1's move (DQN)
#             action_tuple_1 = player.get_Action(environment, player_num="1", events=events, state=state)

#             if action_tuple_1 == (None,) + (None,):
#                 break

#             reward_1, done_1 = environment.move(action_tuple_1, agent_type="DQN", player_num="1")
#             environment.draw_header(done_1, main_surf) #################################
            
#             if done_1:
#                 environment.restart()
#                 break

#             ################## TO REDO OR REMOVE FROM HERE ##############
#             done_2 = False
#             reward_2 = 0

#             next_state = environment.state()


#             #### to not enable to create a new trainer for both DQN agents like in the reversi project ####
#             # if isinstance(player2, Random_Agent) or isinstance(player2, Human_Agent) :
#             action_tuple_2 = player2.get_Action(environment, "2", events)
#             if action_tuple_2 == (None,) + (None,):
#                 break
#             # print("action_tuple_2: ", action_tuple_2, " action_tuple_2[-1]: ", action_tuple_2[-1])
#             reward_2, done_2 = environment.move(action_tuple_2, agent_type="Random_Agent", player_num="2")
#             # print("reward2: ", reward_2, " done_2: ", done_2)
#             # else:
#             #     # For DQN agent
#             #     action_tuple_2 = player.get_Action(environment, player_num="2", events=events, state=state)
#             #     if action_tuple_2 == (None,) + (None,):
#             #         break
#             #     reward_2, done_2 = environment.move(action_tuple_2, agent_type="DQN", player_num="2")
#             #### to not enable to create a new trainer for both DQN agents like in the reversi project until here ####

            
#             # Store experiences in replay buffer for both players
#             buffer.push(state, torch.tensor(action_tuple_1, dtype=torch.int64), torch.tensor(reward_1, dtype=torch.float32),
#                         next_state, torch.tensor(done_1, dtype=torch.float32))

#             ########## to enable??? #####
#             ## Check if action_tuple_2 is not None before pushing to buffer
#             # if action_tuple_2[-1] is not None:  # Ensure it's not None
#             #     buffer.push(state, torch.tensor(action_tuple_2, dtype=torch.int64), torch.tensor(reward_2, dtype=torch.float32),
#             #                 next_state, torch.tensor(done_2, dtype=torch.float32))
#             # else:
#             #     print("Skipping buffer push for Player 2 because action_tuple_2 is None.")
#             ########## to enable??? until here #####


#             ################## TO REDO OR REMOVE UNTIL HERE ##############
            
#             if done_2:
#                 environment.restart()
#                 break  # End the game if any player is done

#             state = next_state

#             environment.draw_header(done_2, main_surf) #################################

#             if len(buffer) < MIN_BUFFER:
#                 #print(f"Buffer size too small: {len(buffer)}")
#                 continue
    
#             ############## Train ################

#             if epoch % 10 != 0: #await training after aquiring enough episodes -- add-on
#                 continue

#             states, actions, rewards, next_states, dones = buffer.sample(batch_size)
#             # Player 1's learning
#             #Q_values = player.Q(states, actions)
#             Q_values = player.DQN(states)
#             # next_actions, Q_hat_Values = player_hat.get_Actions_Values(next_states)
#             Q_hat_Values = player.DQN(next_states)#player_hat.get_Actions_Values(next_states)
#             loss = player.DQN.loss(Q_values, rewards, Q_hat_Values, dones)
#             loss.backward()
#             optim.step()
#             optim.zero_grad()


#             #### to not enable to create a new trainer for both DQN agents like in the reversi project ####
#             # Player 2's learning (if DQN)
#             # if isinstance(player2, DQN_Agent):
#             #     Q_values_2 = player2.Q(states, actions)
#             #     next_actions_2, Q_hat_Values_2 = player2_hat.get_Actions_Values(next_states)
#             #     loss_2 = player2.DQN.loss(Q_values_2, rewards, Q_hat_Values_2, dones)
#             #     loss_2.backward()
#             #     optim.step()
#             #     optim.zero_grad()
#             #### to not enable to create a new trainer for both DQN agents like in the reversi project until here ####



#         if epoch % C == 0:
#             # player_hat.fix_update(dqn=player.DQN)
#             player_hat.DQN.load_state_dict(player.DQN.state_dict())

#         # Print stats after every epoch
#         print(f'Epoch: {epoch} | Loss: {loss:.7f} | LR: {scheduler.get_last_lr()} | Step: {step}')#  | Score: {environment.score} | Best Score: {best_score} 

#         # Save stats every 10 epochs
#         if epoch % 10 == 0:
#             #scores.append(environment.score)
#             losses.append(loss.item())
#             player_1_wins.append(environment.player_1_won)
#             player_2_wins.append(environment.player_2_won)

#         # avg = (avg * (epoch % 10) + environment.score) / (epoch % 10 + 1)
#         if (epoch + 1) % 10 == 0:
#             # avg_score.append(avg)
#             # print(f'Average score over last 10 games: {avg}')
#             # avg = 0
#             wandb.log({
#                 "loss": loss.item(),
#                 "Player 1 Wins": environment.player_1_won,
#                 "Player 2 Wins": environment.player_2_won
#             })

#         # Save checkpoint every 1000 epochs
#         if epoch % 1000 == 0 and epoch > 0:
#             checkpoint = {
#                 'epoch': epoch,
#                 'model_state_dict': player.DQN.state_dict(),
#                 'optimizer_state_dict': optim.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'loss': losses,
#                 'player_1_wins': player_1_wins,
#                 'player_2_wins': player_2_wins,
#                 # 'scores': scores,
#                 # 'avg_score': avg_score
#             }
#             torch.save(checkpoint, checkpoint_path)
#             torch.save(buffer, buffer_path)
#             print(f"Checkpoint saved at epoch {epoch}.")

# if __name__ == "__main__":
#     main()