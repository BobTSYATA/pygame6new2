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



RUN_NUM = 105

path_load= None
path_Save=f'DataTraining/params_{RUN_NUM}.pth'


def main():
    pygame.init()
    environment = Environment()
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    main_surf.fill(LIGHTGRAY) 

    player = DQN_Agent(train=True, player_num=1)

    Q = player.DQN
    Q_hat :DQN = Q.copy()
    Q_hat.train = False


    # Initialize Player 2 with a specific agent type (Human, Random, DQN)
    player2 = Random_Agent()
    # player2 = Baseline_Agent()

    batch_size = 64
    buffer = ReplayBuffer()
    learning_rate = 0.002 
    ephocs = 1000
    start_epoch = 1
    C = 100
    avgLoss = 0
    loss = torch.Tensor([0])
    loss_count = 0

    epsiln_decay = 2000 

    avglosses = []
    

    optim = torch.optim.Adam(Q.parameters(), lr=learning_rate)

    step = 0


    for epoch in range(start_epoch, ephocs):
        environment.restart()

        after_state_2 = None
        state = environment.set_init_state(player_num="1")

        print(f"Epoch {epoch}/{ephocs} starting...")
        done = False



        while not done:
            # print(f"Step: {step}", end='\r')
            step += 1

            events = pygame.event.get()
            for event in events:
                if event.type == pygame.QUIT:
                    return

            environment.draw_header(done, main_surf)
            # Player 1's turn
            action_tuple_1 = player.get_Action(environment, state=state, train=True, epoch=epoch)
            reward1, done = environment.move(epoch,main_surf, action_tuple_1, agent_type="DQN", player_num="1")



            after_state = environment.get_next_state()
            
            if done:
                buffer.push(state, action_tuple_1, reward1, after_state, done) 
                break

            # Player 2's turn
            environment.draw_header(done, main_surf)
            action_tuple_2 = player2.get_Action(environment, "2")
            reward2, done = environment.move(epoch,main_surf, action_tuple_2, agent_type="Random_Agent", player_num="2")


            after_state_2 = environment.get_next_state()
            reward = reward1 + reward2 


            buffer.push(state, action_tuple_1, reward, after_state_2, done)

            state = after_state_2 

            if epoch < batch_size:
                continue
            
            states, actions, rewards, next_states, dones = buffer.sample(batch_size)

            Q_values = Q(states[0], actions)

            next_actions = player.get_actions(environment, next_states, dones)
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
      

        if epoch % C == 0:
            print(f"Updating Q_hat at epoch {epoch}")
            Q_hat.load_state_dict(Q.state_dict())

        step = 0


    player.save_param(path_Save)


if __name__ == "__main__":
    main()

