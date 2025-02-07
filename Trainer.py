
import pygame
import torch
from CONSTANTS import *
from Environment import Environment
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
from Human_Agent import Human_Agent
import os


def save_checkpoint(epoch, player, optim, scheduler, buffer, avglosses, player_1_wins, player_2_wins, checkpoint_path, buffer_path):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': player.DQN.state_dict(),
        'optimizer_state_dict': optim.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'avglosses': avglosses,
        'player_1_wins': player_1_wins,
        'player_2_wins': player_2_wins,
    }

def main():
    pygame.init()
    environment = Environment()
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    main_surf.fill(LIGHTGRAY) #################################

    #best_score = 0

    ####### params ############
    player = DQN_Agent()  # DQN agent for Player 1
    player_hat = DQN_Agent()
    player_hat.DQN = player.DQN.copy()

    # Initialize Player 2 with a specific agent type (Human, Random, DQN)
    player2 = Random_Agent()
    # agent_type2 = "random" #input("Choose Player 2's agent type (Human/Random/DQN): ").lower()
    # if agent_type2 == "random":
    #     player2 = Random_Agent()
    # elif agent_type2 == "human":
    #     player2 = Human_Agent()
    # else:
    #     player2 = DQN_Agent()  # Default to DQN for Player 2
    #     player2_hat = DQN_Agent()
    #     player2_hat.DQN = player2.DQN.copy()  # Copy model for target update

    batch_size = 64
    buffer = ReplayBuffer(path=None)
    learning_rate = 0.1
    ephocs = 1000 #200000
    start_epoch = 0
    C = 9
    avgLoss = 0
    loss = torch.Tensor([0])
    loss_count = 0

    # avgLosses = []
    # avgLoss = 0
    # loss = torch.Tensor([0])
    # loss_count = 0

    player_1_wins, player_2_wins, avglosses = [], [], []

    optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [1000, 2000, 3000, 5000, 7000, 9000], gamma=0.85)
    step = 0

    RUN_NUM = 32

    ######### checkpoint Load ############
    checkpoint_path = f"DataTraining/checkpoint{RUN_NUM}.pth" 
    buffer_path = f"DataTraining/buffer{RUN_NUM}.pth" 
    if os.path.exists(checkpoint_path) and os.path.getsize(checkpoint_path) > 0:
        try:
            checkpoint = torch.load(checkpoint_path)
            start_epoch = checkpoint['epoch'] + 1
            player.DQN.load_state_dict(checkpoint['model_state_dict'])
            player_hat.DQN.load_state_dict(checkpoint['model_state_dict'])
            optim.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            buffer = torch.load(buffer_path)
            avglosses = checkpoint['avglosses']
            player_1_wins = checkpoint['player_1_wins']
            player_2_wins = checkpoint['player_2_wins']
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            start_epoch = 0
            buffer = ReplayBuffer(path=None)
            avglosses, player_1_wins, player_2_wins = [], [], [] 
    else:
        print("No valid checkpoint found, starting fresh.")
        start_epoch = 0
        buffer = ReplayBuffer(path=None)
        avglosses, player_1_wins, player_2_wins = [], [], []
    player.DQN.train()
    player_hat.DQN.eval()


    print(f"Starting training from epoch {start_epoch} to {ephocs}")

    #################################

    for epoch in range(start_epoch, ephocs):
        environment.restart()
        end_of_game = False
        
        
        # Start of epoch
        print(f"Epoch {epoch}/{ephocs} starting...")

        while not end_of_game:
            print(f"Step: {step}", end='\r')
            step += 1

            state = environment.state()

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return

            ############## Sample Environment #########################
            # Player 1's move (DQN)
            action_tuple_1 = player.get_Action(environment, player_num="1", events=events, state=state)

            ####new####
            done_1 = environment.move(action_tuple_1, agent_type="DQN", player_num="1", state=state)


            conquering_troops = environment.update_troop_units()



            for troop in conquering_troops:   
                #print("push: self.id: ",troop.id,"state: ", troop.created_state, "reward: ", troop.reward, "action: ", troop.created_action) 
                # print("Pushing to ReplayBuffer:")
                # print("State:", troop.created_state)
                # print("Action:", troop.created_action)
                # print("Reward:", troop.reward)
                # print("Next State:", troop.next_state)              
                buffer.push(troop.created_state, torch.tensor(troop.created_action, dtype=torch.int64), torch.tensor(troop.reward, dtype=torch.float32),
                    troop.next_state, torch.tensor(troop.done, dtype=torch.float32)) # , torch.tensor(done_1, dtype=torch.float32) 
                ##TO DO:## to make the environment to be for 4 islands -- doing -- to try and make it easy to switch between 4/8 islands, just to comment it out and add a variable etc..
                ##TO DO:## to fix the trainer to be like trainder_wandb -- done
                ##TO DO:## to add done to each troop_unit too and give it at buffer.push -- done, but right now im getting the done when it was sent, 
                # so i need to rethink in the future causeidk if i should do that like that / differently and how to do do it.


            ####new####

            #reward_1, done_1 = environment.move(action_tuple_1, agent_type="DQN", player_num="1", state=state)
            if done_1:
                break


            environment.draw_header(done_1, main_surf)


            action_tuple_2 = player2.get_Action(environment, "2", events)

            # print("action_tuple_2: ", action_tuple_2, " action_tuple_2[-1]: ", action_tuple_2[-1])
            done_2 = environment.move(action_tuple_2, agent_type="Random_Agent", player_num="2",state=state)



            if done_2:
                break  # End the game if any player is done

            environment.draw_header(done_2, main_surf)


            if done_2:
                break  # End the game if any player is done

 

            if len(buffer) < MIN_BUFFER:
                #print(f"Buffer size too small: {len(buffer)}")
                continue
    

            states, actions, rewards, next_states = buffer.sample(batch_size) # , dones
            Q_values = player.DQN(states)
            Q_hat_Values = player_hat.DQN(next_states)
            loss = player.DQN.loss(Q_values, rewards, Q_hat_Values) # , dones
            loss.backward()
            optim.step()
            optim.zero_grad()
            scheduler.step()

            if loss_count <= 1000:
                avgLoss = (avgLoss * loss_count + loss.item()) / (loss_count + 1)
                loss_count += 1
            else:
                avgLoss += (loss.item()-avgLoss)* 0.00001 

         



        if epoch % C == 0:
            # player_hat.fix_update(dqn=player.DQN)
            player_hat.DQN.load_state_dict(player.DQN.state_dict())

        # Print stats after every epoch
        print(f'Epoch: {epoch} | Loss: {loss.item()} | LR: {scheduler.get_last_lr()} | Step: {step}')#  | Score: {environment.score} | Best Score: {best_score} 
        step = 0
        # Save stats every 10 epochs
        if (epoch+1) % 100 == 0:
            avglosses.append(avgLoss)
            print("===>> player_1_wins: ", environment.player_1_won, " player_2_wins: ", environment.player_2_won)

        # Save checkpoint every 1000 epochs
        if epoch % 1000 == 0 and epoch > 0:
            save_checkpoint(epoch, player, optim, scheduler, buffer, avglosses, environment.player_1_won, environment.player_2_won, checkpoint_path, buffer_path)

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
# import os



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
#     # agent_type2 = "random" #input("Choose Player 2's agent type (Human/Random/DQN): ").lower()
#     # if agent_type2 == "random":
#     #     player2 = Random_Agent()
#     # elif agent_type2 == "human":
#     #     player2 = Human_Agent()
#     # else:
#     #     player2 = DQN_Agent()  # Default to DQN for Player 2
#     #     player2_hat = DQN_Agent()
#     #     player2_hat.DQN = player2.DQN.copy()  # Copy model for target update

#     batch_size = 50
#     buffer = ReplayBuffer(path=None)
#     learning_rate = 0.00001
#     ephocs = 100#200000
#     start_epoch = 0
#     C = 3
#     loss = torch.tensor(-1)
#     avg = 0
#     #scores, losses, avg_score = [], [], []
#     losses = []
#     optim = torch.optim.Adam(player.DQN.parameters(), lr=learning_rate)
#     scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, [10000 * 1000], gamma=0.5)
#     step = 0

#     ######### checkpoint Load ############
#     checkpoint_path = "DataTraining/checkpoint1.pth" 
#     buffer_path = "DataTraining/buffer1.pth" 
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
#             #scores = checkpoint['scores']
#             #avg_score = checkpoint['avg_score']
#         except Exception as e:
#             print(f"Error loading checkpoint: {e}")
#             start_epoch = 0
#             buffer = ReplayBuffer(path=None)
#             losses = []
#             #scores = []
#             #avg_score = []
#     else:
#         print("No valid checkpoint found, starting fresh.")
#         start_epoch = 0
#         buffer = ReplayBuffer(path=None)
#         losses = []
#         # scores = []
#         # avg_score = []
#     player.DQN.train()
#     player_hat.DQN.eval()

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

#             # if action_tuple_1 == (None,) + (None,):
#             #     print("(None,) + (None,)")
#             #     break

#             reward_1, done_1 = environment.move(action_tuple_1, agent_type="DQN", player_num="1")
#             environment.draw_header(done_1, main_surf) #################################
            
#             if done_1:
#                 break

#             ################## TO REDO OR REMOVE FROM HERE ##############
#             done_2 = False
#             reward_2 = 0

#             next_state = environment.state()


#             #### to not enable to create a new trainer for both DQN agents like in the reversi project ####
#             # if isinstance(player2, Random_Agent) or isinstance(player2, Human_Agent) :
#             action_tuple_2 = player2.get_Action(environment, "2", events)
#             # if action_tuple_2 == (None,) + (None,):
#             #     print("(None,) + (None,)")
#             #     break
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
#             environment.draw_header(done_2, main_surf)


#             if done_2:
#                 break  # End the game if any player is done

#             state = next_state

#             #environment.draw_header(done_2, main_surf) #################################

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
#         step = 0
#         # Save stats every 10 epochs
#         if epoch % 10 == 0:
#             #scores.append(environment.score)
#             losses.append(loss.item())

#         # avg = (avg * (epoch % 10) + environment.score) / (epoch % 10 + 1)
#         # if (epoch + 1) % 10 == 0:
#         #     avg_score.append(avg)
#         #     print(f'Average score over last 10 games: {avg}')
#         #     avg = 0

#         # Save checkpoint every 1000 epochs
#         if epoch % 1000 == 0 and epoch > 0:
#             checkpoint = {
#                 'epoch': epoch,
#                 'model_state_dict': player.DQN.state_dict(),
#                 'optimizer_state_dict': optim.state_dict(),
#                 'scheduler_state_dict': scheduler.state_dict(),
#                 'loss': losses#,
#                 # 'scores': scores,
#                 # 'avg_score': avg_score
#             }
#             torch.save(checkpoint, checkpoint_path)
#             torch.save(buffer, buffer_path)
#             print(f"Checkpoint saved at epoch {epoch}.")

# if __name__ == "__main__":
#     main()


