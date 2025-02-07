
#Environment.py:
import pygame
import numpy as np
import torch
from CONSTANTS import *
import random
from Island import Island 
from TroopUnit import TroopUnit
import math
from itertools import combinations
from State_ONLY_FOR_CALLING import State_ONLY_FOR_CALLING

class Environment:
    def __init__(self) -> None:
        pygame.init()
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption('State.io')
        self.clock = pygame.time.Clock()


        self.screen.fill((19,19,19))


        self.islands = []  # Initialize the islands attribute here

        # Generate islands
        self.generate_islands()

        self.troop_units = []

        # Create header surface
        self.header_surf = pygame.Surface((WIDTH, 100))
        self.header_surf.fill(BLUE)

        self.last_troop_generation_time = pygame.time.get_ticks()

        self.selected_friendly_islands = None#[]  # Initialize the attribute here

        self.selected_enemy_islands = None#[]  # Initialize the attribute here

        self.EnemyIslandNum = 0
        self.EnemyTroopNum = 0

        self.FriendlyIslandNum = 0
        self.FriendlyTroopNum = 0

        self.should_close_game = False

        self.player_1_won = 0
        self.player_2_won = 0


        #self.Current_Player = 1 # first turn belongs to player number 1 (friendly)
        #self.State = self.state()
        

        #self.score = 0


    


    def write(self, surface, text, pos=(50, 20)):
        font = pygame.font.SysFont("Rubik Mono One", 36)
        text_surface = font.render(text, True, WHITE)
        surface.blit(text_surface, pos)


    def another_game(self):
        while True:
            #print("stuck in loop")
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            return True
            # keys = pygame.key.get_pressed()
            # if keys[pygame.K_y]:
            #     return True
            # if keys[pygame.K_n]:
            #     return False


        

    def draw_header(self, done, main_surf):
        self.header_surf.fill(BLUE)

        if done:
            self.write(self.header_surf, f"End Of Game - Friendly I.N {self.FriendlyIslandNum} - Enemy I.N {self.EnemyIslandNum}", pos=(150, 0))
            self.write(self.header_surf, f"Friendly T.N {self.FriendlyTroopNum} - Enemy T.N {self.EnemyTroopNum}", pos=(150, 30))
            self.write(self.header_surf, "Another Game ?  Y / N", pos=(150, 60)) # 300, 60
            self.screen.blit(self.header_surf, (0, 0))
            pygame.display.update()

    
            if self.another_game():
                self.restart()
                self.reset_troop_units()
            else:
                self.should_close_game = True
        else:
            self.update()
            # main_surf.blit(self.background_image, (0, 0))
            main_surf.fill((19,19,19))
            self.draw(main_surf)
            #print("self.FriendlyIslandNum: ",self.FriendlyIslandNum)
            self.write(self.header_surf, f"Friendly Island Number: {self.FriendlyIslandNum}", (50, 20))
            self.write(self.header_surf, f"Friendly Troop Number: {self.FriendlyTroopNum}", (400, 20))
            self.write(self.header_surf, f"Enemy Island Number: {self.EnemyIslandNum}", (50, 60))
            self.write(self.header_surf, f"Enemy Troop Number: {self.EnemyTroopNum}", (400, 60))
            self.screen.blit(main_surf, (0, 100))
            self.screen.blit(self.header_surf, (0, 0))
            pygame.display.update()
            self.clock.tick(FPS)



    def generate_troops(self):
        TRAINING = True
        if TRAINING:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_troop_generation_time >= 500:# 1000 / 5 = 200#1#100
                for island in self.islands:
                    if island.type in [FRIENDLY, ENEMY] and len(self.troop_units) < MAX_TROOP_UNITS:
                        island.troops += 1
                self.last_troop_generation_time = current_time
        else:
            current_time = pygame.time.get_ticks()
            if current_time - self.last_troop_generation_time >= 1000:
                for island in self.islands:
                    if island.type in [FRIENDLY, ENEMY] and len(self.troop_units) < MAX_TROOP_UNITS:
                        island.troops += 1
                self.last_troop_generation_time = current_time




    def generate_islands(self):
        # Clear existing islands
        self.islands.clear()

        # Define positions for each island in the 3x3 grid
        island_positions = [
            #(WIDTH // 4, MAIN_SURF_HEIGHT // 4),        # Top left
            (WIDTH // 2, MAIN_SURF_HEIGHT // 4),        # Top center
            #(3 * WIDTH // 4, MAIN_SURF_HEIGHT // 4),    # Top right
            (WIDTH // 4, MAIN_SURF_HEIGHT // 2),        # Middle left
            (3 * WIDTH // 4, MAIN_SURF_HEIGHT // 2),    # Middle right
            #(WIDTH // 4, 3 * MAIN_SURF_HEIGHT // 4),    # Bottom left
            (WIDTH // 2, 3 * MAIN_SURF_HEIGHT // 4),    # Bottom center
            #(3 * WIDTH // 4, 3 * MAIN_SURF_HEIGHT // 4) # Bottom right
        ]

        # Generate islands at specified positions
        for x, y in island_positions:
            color = ISLAND_COLORS["neutral"]
            island = Island(x, y, ISLAND_RADIUS, color, NEUTRAL)
            self.islands.append(island)

        # Randomly select one enemy island
        enemy_island = random.choice(self.islands)
        enemy_island.type = ENEMY
        enemy_island.color = ISLAND_COLORS["enemy"]
        self.EnemyIslandNum = 1  # Update count for enemy island

        # Randomly select one friendly island
        friendly_island = random.choice(self.islands)
        while friendly_island == enemy_island:  # Ensure friendly and enemy islands are different
            friendly_island = random.choice(self.islands)
        friendly_island.type = FRIENDLY
        friendly_island.color = ISLAND_COLORS["friendly"]
        self.FriendlyIslandNum = 1  # Update count for friendly island


    def is_end_of_game(self):
        enemy_count = sum(1 for island in self.islands if island.type == ENEMY)
        player_count = sum(1 for island in self.islands if island.type == FRIENDLY)
        
        return enemy_count == 0 or player_count == 0


    def player_won(self, player_num):
        if player_num == "1":
            enemy_count = sum(1 for island in self.islands if island.type == ENEMY)
            return enemy_count == 0
        elif player_num == "2":
            player_count = sum(1 for island in self.islands if island.type == FRIENDLY)
            return player_count == 0


    def reset_troop_units(self):
        self.troop_units = []  # Reset the list of troop units




    def update(self):
        #conquering_troops = self.update_troop_units()
        self.troop_units = [troop_unit for troop_unit in self.troop_units if not troop_unit.has_reached_destination()]
        self.troop_units = self.troop_units[:MAX_TROOP_UNITS]
        self.generate_troops()


    def restart(self):
        self.FriendlyIslandNum = 0
        self.FriendlyTroopNum = 0
        self.EnemyIslandNum = 0
        self.selected_friendly_islands = None #[]
        self.selected_enemy_islands = None #[]
        self.EnemyTroopNum = 0
        #self.score = 0
        self.islands.clear()
        self.generate_islands()
       





    def send_troops(self, to_island, agent_type,player_num):
        if agent_type == "Human_Agent":
            # Calculate the total number of troops to send from the selected islands
            total_troops_to_send = self.selected_friendly_islands.troops
            #if self.EnemyTroopNum + self.FriendlyIslandNum + len(self.selected_enemy_islands)+ len(self.selected_friendly_islands) > MAX_TROOP_UNITS:
            if self.EnemyTroopNum + self.FriendlyTroopNum + 2 > MAX_TROOP_UNITS: # to make sure that for checking  i include the EnemyTroopNum Too unlike in this line and eveything else (after adding random agent) 
                #print("Cannot send troops: exceeds maximum troop units limit. agent_type == Human_Agent")
                # to add UI that says the print too in the middle of the screen - it will be a warning for a few seconds, blinking and then disappearing.
                return None
            #print("agent_type == Human_Agent")
            if total_troops_to_send == 0:
                return None
            # Determine the destination coordinates
            destination = (to_island.x, to_island.y)
            while total_troops_to_send > 0 and self.selected_friendly_islands:
                #self.FriendlyTroopNum += 1
               # print("1 ADDING self.FriendlyTroopNum: ", self.FriendlyTroopNum)
                from_island = self.selected_friendly_islands
                if from_island.troops > 0:
                    # Determine how many troops to send from this island
                    troops_to_send_from_island = min(from_island.troops, total_troops_to_send)
                    if troops_to_send_from_island > 0:  # Only proceed if we are actually sending troops
                        # Create and append troop unit
                        done = self.is_end_of_game()
                        troop_unit = TroopUnit((from_island.x, from_island.y), from_island.color, troops_to_send_from_island, destination)
                        self.troop_units.append(troop_unit)
                        # Update the number of troops left to send
                        total_troops_to_send -= troops_to_send_from_island
                        #print("from_island.troops:", from_island.troops, "troops_to_send_from_island:", troops_to_send_from_island)
                        from_island.troops -= troops_to_send_from_island
                        #print("from_island.troops:", from_island.troops)
                        # Remove the island if no more troops are available
                        if from_island.troops == 0:
                            self.selected_friendly_islands = None #.pop(0)
                else:
                    # Remove the island if it has no troops
                    self.selected_friendly_islands = None #.pop(0)
            # Clear selected islands after sending all available troops
            self.selected_friendly_islands = None #[]
        elif agent_type == "Random_Agent" or agent_type == "DQN":
            if player_num == "1":
                # Calculate the total number of troops to send from the selected islands
                total_troops_to_send = self.selected_friendly_islands.troops
                if self.FriendlyTroopNum + 1 > MAX_TROOP_UNITS / 2:#self.EnemyTroopNum + self.FriendlyTroopNum + 2 > MAX_TROOP_UNITS: # to make sure that for checking  i include the EnemyTroopNum Too unlike in this line and eveything else (after adding random agent) 
                    #print("Cannot send troops: exceeds maximum troop units limit. player_num == 1")
                    return None
                if total_troops_to_send == 0:
                    return None
                destination = (to_island.x, to_island.y)
                while total_troops_to_send > 0 and self.selected_friendly_islands:
                    from_island = self.selected_friendly_islands
                    if from_island.troops > 0:
                        troops_to_send_from_island = min(from_island.troops, total_troops_to_send)
                        if troops_to_send_from_island > 0:
                            done = self.is_end_of_game()
                            troop_unit = TroopUnit((from_island.x, from_island.y), from_island.color, troops_to_send_from_island, destination)
                            self.troop_units.append(troop_unit)
                            total_troops_to_send -= troops_to_send_from_island
                            from_island.troops -= troops_to_send_from_island
                            if from_island.troops == 0:
                                self.selected_friendly_islands = None #.pop(0)
                    else:
                        # Remove the island if it has no troops
                        self.selected_friendly_islands = None #.pop(0)
                # Clear selected islands after sending all available troops
                self.selected_friendly_islands = None #[]
            elif player_num == "2":
                # Calculate the total number of troops to send from the selected islands
                total_troops_to_send = self.selected_enemy_islands.troops
                if self.EnemyTroopNum + 1 > MAX_TROOP_UNITS / 2:#self.EnemyTroopNum + self.FriendlyTroopNum + 2 > MAX_TROOP_UNITS: # to make sure that for checking  i include the EnemyTroopNum Too unlike in this line and eveything else (after adding random agent) 
                    #print("Cannot send troops: exceeds maximum troop units limit. player_num == 2")
                    return None
                if total_troops_to_send == 0:
                    return None
                destination = (to_island.x, to_island.y)
                while total_troops_to_send > 0 and self.selected_enemy_islands:
                    from_island = self.selected_enemy_islands
                    if from_island.troops > 0:
                        troops_to_send_from_island = min(from_island.troops, total_troops_to_send)
                        if troops_to_send_from_island > 0: 
                            done = self.is_end_of_game()
                            troop_unit = TroopUnit((from_island.x, from_island.y), from_island.color, troops_to_send_from_island, destination)
                            self.troop_units.append(troop_unit)
                            total_troops_to_send -= troops_to_send_from_island
                            from_island.troops -= troops_to_send_from_island
                            if from_island.troops == 0:
                                self.selected_enemy_islands = None# .pop(0)
                    else:
                        # Remove the island if it has no troops
                        self.selected_enemy_islands = None #.pop(0)
                # Clear selected islands after sending all available troops
                self.selected_enemy_islands = None #[]



    def set_init_state (self):
        self.state = State_ONLY_FOR_CALLING()
        self.state.legal_actions = self.all_possible_actions_for_agent(self.state.player)#self.get_all_legal_Actions(self.state)
        return self.state
    
    def next_state(self, state: State_ONLY_FOR_CALLING):
        next_state = state.copy()
        next_state.legal_actions = self.all_possible_actions_for_agent(self.state.player)#self.get_all_legal_Actions(self.state)
        next_state.switch_players()
        return self.state




    def update_troop_units(self):
        reached_destination = []
        #given_reward = 0 # to make this a list? BLANK SHEET TO RETURN TO THIS for now to ask someone and think about it
        #conquering_troops = []

        enemy_player_conquered_island_amount = 0
        friendly_player_conquered_island_amount = 0
        for troop_unit in self.troop_units:
            dx = troop_unit.destination[0] - (troop_unit.rect.x + troop_unit.rect.width / 2)
            dy = troop_unit.destination[1] - (troop_unit.rect.y + troop_unit.rect.height / 2)
            distance = max(1, math.sqrt(dx ** 2 + dy ** 2))
            dx /= distance
            dy /= distance
            troop_unit.rect.x += dx * TROOP_UNIT_SPEED
            troop_unit.rect.y += dy * TROOP_UNIT_SPEED

            if troop_unit.has_reached_destination():
                reached_destination.append(troop_unit)

        for troop_unit in reached_destination:
            destination_island = next((island for island in self.islands
                                    if math.sqrt((island.x - troop_unit.destination[0]) ** 2 +
                                                (island.y - troop_unit.destination[1]) ** 2) <= ISLAND_RADIUS), None)
            if destination_island:
                # Check troop type by its color to determine if it's friendly or enemy
                if troop_unit.color == ISLAND_COLORS["friendly"]:  # Friendly troop
                    if destination_island.type == FRIENDLY:
                        
                        #print(f"AAA troop.id: {troop_unit.id}")
                        # troop_unit.reward = 0

                        # conquering_troops.append(troop_unit)


                        destination_island.troops += troop_unit.troop_count
                        
                    elif destination_island.type in [NEUTRAL, ENEMY]:
                        if troop_unit.troop_count > destination_island.troops:

                            # only adding conquering_troops units + given_reward for friendly for now to ask someone and think about it
                            #given_reward += 10 # BLANK SHEET TO RETURN TO THIS for now to ask someone and think about it
                            # if destination_island.type == ENEMY:
                            #     troop_unit.reward = 100
                            # elif destination_island.type == NEUTRAL:
                            #     troop_unit.reward = 10
                         #   print(f"BBB troop.id: {troop_unit.id}")

                            #done = self.is_end_of_game()
                            # troop_unit.reward = 1
                            # conquering_troops.append(troop_unit)

                            
                            destination_island.type = FRIENDLY
                            destination_island.color = ISLAND_COLORS["friendly"]
                            destination_island.troops = troop_unit.troop_count - destination_island.troops
                            self.FriendlyIslandNum += 1
                            friendly_player_conquered_island_amount += 1
                            #print("conquer: self.id: ",troop_unit.id,"state: ", troop_unit.created_state, "reward: ", troop_unit.reward, "action: ", troop_unit.created_action)
                        else:
                       #     print(f"CCC troop.id: {troop_unit.id}")

                            # troop_unit.reward = 0#-1
                            # conquering_troops.append(troop_unit)

                            destination_island.troops -= troop_unit.troop_count
                            #print("not conquer: self.id: ",troop_unit.id,"state: ", troop_unit.created_state, "reward: ", troop_unit.reward, "action: ", troop_unit.created_action)
                elif troop_unit.color == ISLAND_COLORS["enemy"]:  # Enemy troop
                    if destination_island.type == ENEMY:

                        # troop_unit.reward = 0
                        # conquering_troops.append(troop_unit)

                        destination_island.troops += troop_unit.troop_count
                    elif destination_island.type in [NEUTRAL, FRIENDLY]:
                        if troop_unit.troop_count > destination_island.troops:
                            
                            # troop_unit.reward = -1
                            # conquering_troops.append(troop_unit)

                            destination_island.type = ENEMY
                            destination_island.color = ISLAND_COLORS["enemy"]
                            destination_island.troops = troop_unit.troop_count - destination_island.troops
                            self.EnemyIslandNum += 1
                            enemy_player_conquered_island_amount += 1
                        else:
                            # troop_unit.reward = 0
                            # conquering_troops.append(troop_unit)
                            destination_island.troops -= troop_unit.troop_count

                # Remove the troop unit after it reaches its destination
                self.troop_units.remove(troop_unit)
        self.troop_units = [troop_unit for troop_unit in self.troop_units if not troop_unit.has_reached_destination()]
        self.troop_units = self.troop_units[:MAX_TROOP_UNITS]

        # Update troop counts
        self.update_FAndE_Troo_Num()
        #print("friendly_player_conquered_island_amount: ", friendly_player_conquered_island_amount, " enemy_player_conquered_island_amount: ", enemy_player_conquered_island_amount)
        # print(f"len(conquering_troops) = {len(conquering_troops)}, conquering_troops: {conquering_troops}")
        #return conquering_troops#given_reward, conquering_troops
        #return friendly_player_conquered_island_amount, enemy_player_conquered_island_amount



    def update_FAndE_Troo_Num(self):
        friendlyTroopsNum = 0
        enemyTroopsNum = 0
        for troop_unit in self.troop_units:
            if troop_unit.color == ISLAND_COLORS["friendly"]:
                friendlyTroopsNum += 1
            elif troop_unit.color == ISLAND_COLORS["enemy"]:
                enemyTroopsNum += 1
        self.FriendlyTroopNum = friendlyTroopsNum
        self.EnemyTroopNum = enemyTroopsNum






    def get_clicked_island(self):
        mouse_pos = pygame.mouse.get_pos()
        adjusted_mouse_pos = (mouse_pos[0], mouse_pos[1] - 100)
        #print("self.islands: ", self.islands)
        for island in self.islands:
            distance = math.sqrt((island.x - adjusted_mouse_pos[0]) ** 2 + (island.y - adjusted_mouse_pos[1]) ** 2)
            if distance <= ISLAND_RADIUS:
                return island
        return None



    def all_possible_actions_for_agent(self, agentType):
        """
        Returns all possible actions for the agent.
        Each action consists of a source island and a destination island.

        Args:
            agentType (int): The type of the agent (1 for friendly player, 2 for enemy player).

        Returns:
            list of tuples: Each tuple represents an action (source_index, destination_index).
        """
        all_possible_actions = []

        if agentType == 1:  # Friendly player
            all_my_islands = self.get_all_friendly_islands()  # Friendly islands
        elif agentType == 2:  # Enemy player
            all_my_islands = self.get_all_enemy_islands()  # Enemy islands
        else:
            return all_possible_actions  # Invalid agent type

        all_islands = self.get_all_islands()  # All islands (including neutral/enemy)

        for source_island in all_my_islands:
            source_index = self.get_island_index(source_island)
            if source_index is not None:
                for destination_island in all_islands:
                    destination_index = self.get_island_index(destination_island)
                    if destination_index is not None:
                        # Create an action as a tuple (source_index,) + (destination_index,)
                        action = (source_index,) + (destination_index,)
                        all_possible_actions.append(action)
                    else:
                        # Handle invalid destination or self-targeting
                        action = (source_index,) + (None,)
                        all_possible_actions.append(action)
        #print("all_possible_actions: ", all_possible_actions, "length: " ,len(all_possible_actions))
        return all_possible_actions

    







    def draw_troop_units(self, main_surf):
        for troop_unit in self.troop_units:
            troop_unit.draw(main_surf)


    def draw(self, main_surf):

         # Draw the background image
        # main_surf.blit(self.background_image, (0, 0))
        main_surf.fill((19,19,19))

        #main_surf.fill(BACKGROUND_COLOR)
        for island in self.islands:
            island.draw(main_surf)  # Draw islands onto main_surf
        self.draw_troop_units(main_surf)  # Draw troop units onto main_surf

    def get_island_index(self, clicked_island):
        # Assuming 'clicked_island' is an Island object, we find its index in the 'self.islands' list.
        try:
            return self.islands.index(clicked_island)
        except ValueError:
            return None  # If the island is not found, return None

            
    def move(self, action_tuple, agent_type, player_num, state):
        reward = 0
        done = False

        
        #print("left_clicked_selected_islands: ", left_clicked_selected_islands, " . right_clicked_selected_islands: ", right_clicked_selected_islands, " . agent_type: ", agent_type, ".")
        left_clicked_selected_islands = action_tuple[0]
        right_clicked_selected_islands = action_tuple[1]

        attacking_island_index = right_clicked_selected_islands
        #selected_island_indices = left_clicked_selected_islands# [i for i, flag in enumerate(left_clicked_selected_islands) if flag == 1]
        attacking_island = self.islands[attacking_island_index] if attacking_island_index is not None else None
        self.selected_friendly_islands = self.islands[left_clicked_selected_islands] if attacking_island_index is not None else None # [self.islands[i] for i in selected_island_indices]
        self.selected_enemy_islands = self.islands[left_clicked_selected_islands] if attacking_island_index is not None else None # [self.islands[i] for i in selected_island_indices]

        if attacking_island:
           # print(f"Attacking island type: {attacking_island.type}")

            if agent_type == "Human_Agent":
                self.send_troops(attacking_island,agent_type,player_num)
                #if val != None:
                # if self.selected_friendly_islands:
                #     total_troops = self.selected_friendly_islands.troops
                #     if total_troops > 0:
                #         if attacking_island.type in [NEUTRAL, ENEMY]:
                #             defending_troops = attacking_island.troops
                #             if total_troops > defending_troops:
                #                 #print("agent_type == Human_Agent agent_type == Human_Agent agent_type == Human_Agent")
                #                 leftover_troops = total_troops - defending_troops
                #         elif attacking_island.type == FRIENDLY:
                #             #print("agent_type == Human_Agent agent_type == Human_Agent agent_type == Human_Agent")
                #             attacking_island.troops += total_troops

                #                 # can add and check in the future if it has any enemy troops heading to it and if so to add reward to it ############### TO DO
                # self.selected_friendly_islands = None #[]
            elif agent_type == "Random_Agent" or agent_type == "DQN":
                if player_num == "1":
                    #if val != None:
                    if self.selected_friendly_islands:
                        total_troops = self.selected_friendly_islands.troops
                        if total_troops > 0:
                            if attacking_island.type in [NEUTRAL, ENEMY]: # NEUTRAL, FRIENDLY
                                defending_troops = attacking_island.troops
                                #if attacking_island.type == ENEMY:
                                #reward += 0.1#0.2 to do: because i changed it to 0.1 it will be negative if 2 conquer at the same time so the avgloss might go up to search
                                if total_troops > defending_troops:
                                    #leftover_troops = total_troops - defending_troops
                                    if attacking_island.type == NEUTRAL:
                                        reward += 0.6  # Positive reward for POSSIBLE conquering
                                    if attacking_island.type == ENEMY:
                                        reward += 1.8  # Positive reward for POSSIBLE conquering  
                            # elif attacking_island.type == FRIENDLY:
                            #     #attacking_island.troops += total_troops
                            #     reward += 0.1 # Positive reward for POSSIBLE reinforcement  

                    self.send_troops(attacking_island,agent_type,player_num)

                elif player_num == "2":
                    
                    #if val != None:
                    if self.selected_enemy_islands:
                        total_troops = self.selected_enemy_islands.troops #sum(island.troops for island in self.selected_enemy_islands)
                        if total_troops > 0:
                            if attacking_island.type in [NEUTRAL, FRIENDLY]:
                                defending_troops = attacking_island.troops
                                if total_troops > defending_troops:
                                    # leftover_troops = total_troops - defending_troops
                                    if attacking_island.type == NEUTRAL:
                                        reward -= 0.6 #0.6 maybe to lower the 0.6 because it can't control this?
                                    if attacking_island.type == FRIENDLY:
                                        reward -= 1.8
                            # elif attacking_island.type == ENEMY:
                                # attacking_island.troops += total_troops
                    #self.selected_enemy_islands = None #[]
                    self.send_troops(attacking_island,agent_type,player_num)

        # else:
        #     #print("No attacking island selected.")
        
        self.EnemyIslandNum = sum(1 for island in self.islands if island.type == ENEMY)
        self.FriendlyIslandNum = sum(1 for island in self.islands if island.type == FRIENDLY)
        self.update_troop_units()
      
        # # Calculate troop differences
        # total_friendly_troops = sum(island.troops for island in self.islands if island.type == FRIENDLY)
        # total_enemy_troops = sum(island.troops for island in self.islands if island.type == ENEMY)
        # max_troops = MAX_TROOP_UNITS / 2  # Max troops per player

        

        # # Reward based on island control
        # island_reward = self.FriendlyIslandNum - self.EnemyIslandNum

        # # Reward based on troop strength
        # troop_reward = (total_friendly_troops - total_enemy_troops) / max_troops

        # # Combine rewards with weights
        # w1, w2 = 1.0, 0.5  # Weights for islands and troops, respectively
        # reward += w1 * island_reward + w2 * troop_reward
        # #print("reward before done: ", reward)

        done = self.is_end_of_game()
        
        if done:
            player_won =  self.player_won("1")
            if player_won:
                reward += 100
                self.player_1_won += 1
            else:
                reward -= 100
                self.player_2_won += 1

        state.switch_players()

        # overall_friendly_troops_num = 0 # maybe to implement somethign similar later
        # overall_enemy_troops_num = 0
        # friendlyTroopsNum = 0
        # enemyTroopsNum = 0
        # for troop_unit in self.troop_units:
        #     if troop_unit.color == ISLAND_COLORS["friendly"]:
        #         friendlyTroopsNum += troop_unit.troop_count
        #     elif troop_unit.color == ISLAND_COLORS["enemy"]:
        #         enemyTroopsNum += troop_unit.troop_count
        # friendlyIslandTroopCount = 0
        # enemyIslandTroopCount = 0
        # friendlyIslandCount = len(self.get_all_friendly_islands())
        # enemyIslandCount = len(self.get_all_enemy_islands())
        # for island in self.islands:
        #     if island.color == ISLAND_COLORS["friendly"]:
        #         friendlyIslandTroopCount += island.troops
        #     elif island.color == ISLAND_COLORS["enemy"]:
        #         enemyIslandTroopCount += island.troops
        # overall_friendly_troops_num = friendlyTroopsNum + friendlyIslandTroopCount
        # overall_enemy_troops_num = enemyTroopsNum + enemyIslandTroopCount
        # reward += (overall_friendly_troops_num + (1.5 * friendlyIslandCount)) - (overall_enemy_troops_num + (1.5 * enemyIslandCount))  #(overall friendly troops num(in troopunits + isalnds) + friendly islands num (meaning the troops that will be generated a sec)) - (enemy overall troops num + enemy islands num) so it will try to have more troop units all the time? 


        # if player_num == "1":
        #     self.Current_Player = 1
        # else:
        #     self.Current_Player = 2
        # if done:
        #     #print("end of game")
        #     player_won =  self.player_won(player_num)
        #     if player_won:
        #         reward += 100
        #         #print("reward: ", reward)
        #     else:
        #         reward -= 100
        #         #print("reward: ", reward)
        #     print("reward done: ", reward)
        # if done:
        #     reward -= 3 
        # print("rrrrrrrrrrrreward : ", reward)
        return reward, done # reward, done





    def legal(self, action, agentType):
        possible_actions = self.get_possible_actions(action,agentType)

        # Extract the click type from the action
        click_type = action.split("_")[0]

        # Check if the action with the same click type and coordinates is present in possible actions
        for possible_action in possible_actions:
            if possible_action.startswith(click_type):
                return True
        return False



    def get_possible_actions(self, action, agentType):

        if agentType == FRIENDLY:
            clicked_island = self.get_clicked_island()
            possible_actions = []

            if action == "LEFT_CLICK":
                if clicked_island and clicked_island.type == FRIENDLY:
                    # If it's a left-click on a friendly island, list all friendly islands
                    for island in self.islands:
                        if island.type == FRIENDLY:
                            possible_actions.append(f"LEFT_CLICK_{island.x}_{island.y}")
            elif action == "RIGHT_CLICK":
                if clicked_island:
                    # If it's a right-click on any island, list all islands
                    for island in self.islands:
                        possible_actions.append(f"RIGHT_CLICK_{island.x}_{island.y}")

            #print(possible_actions)
            return possible_actions
        else: # unneeded get possible actions only for human agent rn and legal too
            clicked_island = self.get_clicked_island()
            possible_actions = []

            if action == "LEFT_CLICK":
                if clicked_island and clicked_island.type == ENEMY:
                    # If it's a left-click on a friendly island, list all friendly islands
                    for island in self.islands:
                        if island.type == ENEMY:
                            possible_actions.append(f"LEFT_CLICK_{island.x}_{island.y}")
            elif action == "RIGHT_CLICK":
                if clicked_island:
                    # If it's a right-click on any island, list all islands
                    for island in self.islands:
                        possible_actions.append(f"RIGHT_CLICK_{island.x}_{island.y}")

            #print(possible_actions)
            return possible_actions


    # Assuming this is part of a class definition (like Environment)


    def get_all_islands(self):
        return self.islands
    
    def get_all_enemy_islands(self):
        return [island for island in self.islands if island.type == ENEMY]
    
    def get_all_friendly_islands(self):
        return [island for island in self.islands if island.type == FRIENDLY]


    def initialize_state_tensor(self):
        return torch.zeros(STATE_SIZE)


    # def state(self):
    #     #def set_graphics_staste():
    #     # Initialize state tensor
    #     state_tensor = self.initialize_state_tensor()

    #     # Number of elements per troop unit
    #     ELEMENTS_PER_TROOP_UNIT = 6

    #     # Fill in the troop units data
    #     for i, troop_unit in enumerate(self.troop_units[:MAX_TROOP_UNITS]):
    #         index = i * ELEMENTS_PER_TROOP_UNIT
    #         state_tensor[index] = troop_unit.rect.centerx
    #         state_tensor[index + 1] = troop_unit.rect.centery
    #         state_tensor[index + 2] = troop_unit.troop_count
    #         state_tensor[index + 3] = troop_unit.type  # R
    #         state_tensor[index + 4] = troop_unit.destination[0]  # x-coordinate of destination
    #         state_tensor[index + 5] = troop_unit.destination[1]  # y-coordinate of destination

    #     # Number of elements per island
    #     ELEMENTS_PER_ISLAND = 4

    #     # Start index for islands data
    #     island_start_index = MAX_TROOP_UNITS * ELEMENTS_PER_TROOP_UNIT

    #     # Fill in the islands data
    #     for i, island in enumerate(self.islands[:MAX_ISLANDS]):
    #         index = island_start_index + i * ELEMENTS_PER_ISLAND
    #         state_tensor[index] = island.x
    #         state_tensor[index + 1] = island.y
    #         state_tensor[index + 2] = island.type  # Assuming type is a numerical representation
    #         state_tensor[index + 3] = island.troops

    #     # Start index for additional variables
    #     # additional_vars_index = island_start_index + MAX_ISLANDS * ELEMENTS_PER_ISLAND

    #     # Fill in additional variables
    #     # state_tensor[additional_vars_index] = self.FriendlyIslandNum
    #     # state_tensor[additional_vars_index + 1] = self.FriendlyTroopNum
    #     # state_tensor[additional_vars_index + 2] = self.EnemyIslandNum
    #     # state_tensor[additional_vars_index + 3] = self.EnemyTroopNum

    #     return state_tensor
    #     #self.State = set_graphics_staste()
    #     # print("self.Current_Player:",self.Current_Player)
    #     #actions = self.all_possible_actions_for_agent(self.Current_Player) # to check Current_Player and the whole new state in env thing is working
    #     #self.State.legal_actions = actions#self.get_all_legal_Actions(self.State)
    #     # print("self.State: ", self.State)
    #     #return self.State