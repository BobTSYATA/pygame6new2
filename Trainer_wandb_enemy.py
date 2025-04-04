import pygame
import torch
from CONSTANTS import *
from Environment import Environment
from DQN_Agent import DQN_Agent
from ReplayBuffer import ReplayBuffer
from Random_Agent import Random_Agent
from DQN import DQN
import wandb
from Tester import Tester

# environment script line 623  

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

RUN_NUM = 126 

path_load= None
path_Save=f'DataTraining/params_{RUN_NUM}.pth'
path_best = f'DataTraining/best_params_{RUN_NUM}.pth'
buffer_path = f'DataTraining/buffer_{RUN_NUM}.pth'
results_path=f'DataTraining/results_{RUN_NUM}.pth'
random_results_path = f'DataTraining/random_results_{RUN_NUM}.pth'
path_best_random = f'DataTraining/best_random_params_{RUN_NUM}.pth'

def main():
    pygame.init()
    environment = Environment()
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    main_surf.fill(LIGHTGRAY) 

    player2 = DQN_Agent(train=True, player_num=2) 
    Q = player2.DQN
    Q_hat :DQN = Q.copy()
    Q_hat.train = False


    # Initialize Player 2 with a specific agent type (Human, Random, DQN)
    player = Random_Agent()
    # player2 = Baseline_Agent()

    batch_size = 64
    buffer = ReplayBuffer()
    learning_rate = 0.002 
    ephocs = 50000
    start_epoch = 1
    C = 100
    avgLoss = 0
    loss = torch.Tensor([0])
    loss_count = 0
    tester = Tester(player1=Random_Agent(), player2=player, env=environment,main_surf=main_surf) 
    tester_fix = Tester(player1=player2, player2=player, env=environment,main_surf=main_surf)
    random_results = []
    results = []
    best_random = -100
    res = 0
    best_res = -200
    epsiln_decay = 2000 
    avglosses = []
    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)
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
        state = environment.set_init_state(player_num="1")

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

            action_tuple_1 = player.get_Action(environment, "1")
            reward1, done = environment.move(epoch,main_surf, action_tuple_1, agent_type="Random_Agent", player_num="1")

            # Record data for Player 1
            player_1_actions.append(action_tuple_1)
            player_1_rewards.append(reward1)
            next_next_states_graphics.append(after_state.Graphics)

            after_state_2 = environment.get_next_state(player_num="2")
            reward = reward1 + reward2 


            buffer.push(state, action_tuple_2, reward, after_state_2, done)

            state = after_state_2



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
                

        if (epoch+1) % 500 == 0:
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
            player2.save_param(path_Save)
            torch.save(random_results, random_results_path)


    player2.save_param(checkpoint_path)

    torch.save({'epoch': epoch, 'results': results, 'avglosses':avglosses}, results_path)
    torch.save(random_results, random_results_path)


if __name__ == "__main__":
    main()
