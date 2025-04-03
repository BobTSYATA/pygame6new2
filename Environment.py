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

    


    def write(self, surface, text, pos=(50, 20)):
        font = pygame.font.SysFont("Rubik Mono One", 36)
        text_surface = font.render(text, True, WHITE)
        surface.blit(text_surface, pos)


    def another_game(self):
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return False
            keys = pygame.key.get_pressed()
            if keys[pygame.K_y]:
                return True
            if keys[pygame.K_n]:
                return False


        

    def draw_header(self, done, main_surf):
        self.header_surf.fill(BLUE)

        if done:
            print("done = ",done)
            self.write(self.header_surf, f"End Of Game - Friendly I.N {self.FriendlyIslandNum} - Enemy I.N {self.EnemyIslandNum}", pos=(150, 0))
            self.write(self.header_surf, f"Friendly T.N {self.FriendlyTroopNum} - Enemy T.N {self.EnemyTroopNum}", pos=(150, 30))
            self.write(self.header_surf, "Another Game ?  Y / N", pos=(150, 60)) 
            self.screen.blit(self.header_surf, (0, 0))
            pygame.display.update()

    
            if self.another_game():
                self.restart()
                self.reset_troop_units()
            else:
                self.should_close_game = True
        else:
            self.update()
            main_surf.fill((19,19,19))
            self.draw(main_surf)
            self.write(self.header_surf, f"Friendly Island Number: {self.FriendlyIslandNum}", (50, 20))
            self.write(self.header_surf, f"Friendly Troop Number: {self.FriendlyTroopNum}", (400, 20))
            self.write(self.header_surf, f"Enemy Island Number: {self.EnemyIslandNum}", (50, 60))
            self.write(self.header_surf, f"Enemy Troop Number: {self.EnemyTroopNum}", (400, 60))
            self.screen.blit(main_surf, (0, 100))
            self.screen.blit(self.header_surf, (0, 0))
            pygame.display.update()
            self.clock.tick(FPS)



    def generate_troops(self):
        for island in self.islands:
            if island.type in [FRIENDLY, ENEMY] and len(self.troop_units) < MAX_TROOP_UNITS:
                island.troops += 1

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
            island = Island(x, y, ISLAND_RADIUS, color, NEUTRAL, START_TROOPS-6)
            self.islands.append(island)

        # Randomly select one enemy island
        enemy_island = random.choice(self.islands)
        enemy_island.type = ENEMY
        enemy_island.color = ISLAND_COLORS["enemy"]
        enemy_island.troops = START_TROOPS
        

        self.EnemyIslandNum = 1  # Update count for enemy island

        # Randomly select one friendly island
        friendly_island = random.choice(self.islands)
        while friendly_island == enemy_island:  # Ensure friendly and enemy islands are different
            friendly_island = random.choice(self.islands)
        friendly_island.type = FRIENDLY
        friendly_island.color = ISLAND_COLORS["friendly"]
        friendly_island.troops = START_TROOPS


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
        self.troop_units = [troop_unit for troop_unit in self.troop_units if not troop_unit.has_reached_destination()]
        self.troop_units = self.troop_units[:MAX_TROOP_UNITS]
        self.generate_troops()


    def restart(self):
        self.FriendlyIslandNum = 0
        self.FriendlyTroopNum = 0
        self.EnemyIslandNum = 0
        self.selected_friendly_islands = None 
        self.selected_enemy_islands = None 
        self.EnemyTroopNum = 0
        self.islands.clear()
        self.generate_islands()
       





    def send_troops(self, to_island, agent_type,player_num):
        if agent_type == "Human_Agent":
            # Calculate the total number of troops to send from the selected islands
            total_troops_to_send = self.selected_friendly_islands.troops
            if self.EnemyTroopNum + self.FriendlyTroopNum + 2 > MAX_TROOP_UNITS: 
                return None
            if total_troops_to_send == 0:
                return None
            # Determine the destination coordinates
            destination = (to_island.x, to_island.y)
            while total_troops_to_send > 0 and self.selected_friendly_islands:
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
                        from_island.troops -= troops_to_send_from_island
                        # Remove the island if no more troops are available
                        if from_island.troops == 0:
                            self.selected_friendly_islands = None 
                else:
                    # Remove the island if it has no troops
                    self.selected_friendly_islands = None 
            # Clear selected islands after sending all available troops
            self.selected_friendly_islands = None 
        elif agent_type == "Random_Agent" or agent_type == "DQN":
            if player_num == "1":
                # Calculate the total number of troops to send from the selected islands
                total_troops_to_send = self.selected_friendly_islands.troops
                if self.FriendlyTroopNum > MAX_TROOP_UNITS / 2:
                    print("Cannot send troops: exceeds maximum troop units limit. player_num == 1")
                    return None
                if total_troops_to_send == 0:
                    print("total_troops_to_send == 0 NEVER SUPPOSED TO HAPPEN SUPPOSED TO GENERATE A TROOP UNIT BEFORE A TURN OR SMT LIKE THAT player_num == 1")
                    return None
                destination = (to_island.x, to_island.y)
                while total_troops_to_send > 0 and self.selected_friendly_islands:
                    from_island = self.selected_friendly_islands
                    if from_island.troops > 0:
                        troops_to_send_from_island = min(from_island.troops, total_troops_to_send)
                        if troops_to_send_from_island > 0:

                            troop_unit = TroopUnit((from_island.x, from_island.y), from_island.color, troops_to_send_from_island, destination)
                            self.troop_units.append(troop_unit)

                            total_troops_to_send -= troops_to_send_from_island
                            from_island.troops -= troops_to_send_from_island
                            if from_island.troops == 0:
                                self.selected_friendly_islands = None 
                            return troop_unit
                    else:
                        # Remove the island if it has no troops
                        self.selected_friendly_islands = None 
                # Clear selected islands after sending all available troops
                self.selected_friendly_islands = None 
            elif player_num == "2":
                # Calculate the total number of troops to send from the selected islands
                total_troops_to_send = self.selected_enemy_islands.troops
                if self.EnemyTroopNum > MAX_TROOP_UNITS / 2:
                    print("Cannot send troops: exceeds maximum troop units limit. player_num == 2")
                    return None
                if total_troops_to_send == 0:
                    print("total_troops_to_send == 0 NEVER SUPPOSED TO HAPPEN SUPPOSED TO GENERATE A TROOP UNIT BEFORE A TURN OR SMT LIKE THAT player_num == 2")
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
                                self.selected_enemy_islands = None
                            return troop_unit
                    else:
                        # Remove the island if it has no troops
                        self.selected_enemy_islands = None 
                # Clear selected islands after sending all available troops
                self.selected_enemy_islands = None 



    def set_init_state (self, player_num):
        num = None
        if player_num == "1":
            num = 1
        elif player_num == "2":
            num = 2
        self.state = State_ONLY_FOR_CALLING(environment=self)
        self.state.legal_actions = self.all_possible_actions_for_agent(num)
        return self.state



    def running_while_troop_inbound(self, done, main_surf):
        self.header_surf.fill(BLUE)

        if done:
            self.write(self.header_surf, f"End Of Game - Friendly I.N {self.FriendlyIslandNum} - Enemy I.N {self.EnemyIslandNum}", pos=(150, 0))
            self.write(self.header_surf, f"Friendly T.N {self.FriendlyTroopNum} - Enemy T.N {self.EnemyTroopNum}", pos=(150, 30))
            self.write(self.header_surf, "Another Game ?  Y / N", pos=(150, 60)) 
            self.screen.blit(self.header_surf, (0, 0))
            pygame.display.update()

    
            if self.another_game():
                self.restart()
                self.reset_troop_units()
            else:
                self.should_close_game = True
        else:
            main_surf.fill((19,19,19))
            self.draw(main_surf)
            self.write(self.header_surf, f"Friendly Island Number: {self.FriendlyIslandNum}", (50, 20))
            self.write(self.header_surf, f"Friendly Troop Number: {self.FriendlyTroopNum}", (400, 20))
            self.write(self.header_surf, f"Enemy Island Number: {self.EnemyIslandNum}", (50, 60))
            self.write(self.header_surf, f"Enemy Troop Number: {self.EnemyTroopNum}", (400, 60))
            self.screen.blit(main_surf, (0, 100))
            self.screen.blit(self.header_surf, (0, 0))
            pygame.display.update()
            self.clock.tick(FPS)


    def resolve_troop_arrival(self, troop_unit):
        """Handles what happens when a troop reaches an island."""
        destination_island = next((island for island in self.islands
                                    if math.sqrt((island.x - troop_unit.destination[0]) ** 2 +
                                                (island.y - troop_unit.destination[1]) ** 2) <= ISLAND_RADIUS), None)
        if destination_island:
            # Check troop type by its color to determine if it's friendly or enemy
            boolean = troop_unit.color == ISLAND_COLORS["friendly"]
            if troop_unit.color == ISLAND_COLORS["friendly"]:  # Friendly troop
                # print("friendly troop unit arrived to destination")
                if destination_island.type == FRIENDLY:
                    # print("friendly troop unit to friendly island ADDING TROOPS")
                    destination_island.troops += troop_unit.troop_count
                elif destination_island.type in [NEUTRAL, ENEMY]:
                    if troop_unit.troop_count > destination_island.troops:
                        # print("friendly troop unit to friendly island CONQUERING")
                        destination_island.type = FRIENDLY
                        destination_island.color = ISLAND_COLORS["friendly"]
                        destination_island.troops = troop_unit.troop_count - destination_island.troops
                        self.FriendlyIslandNum += 1
                    else:
                        # print("friendly troop unit to friendly island NEGATING TROOPS")
                        destination_island.troops -= troop_unit.troop_count
            elif troop_unit.color == ISLAND_COLORS["enemy"]:  # Enemy troop
                if destination_island.type == ENEMY:
                    destination_island.troops += troop_unit.troop_count
                elif destination_island.type in [NEUTRAL, FRIENDLY]:
                    if troop_unit.troop_count > destination_island.troops:                    
                        destination_island.type = ENEMY
                        destination_island.color = ISLAND_COLORS["enemy"]
                        destination_island.troops = troop_unit.troop_count - destination_island.troops
                        self.EnemyIslandNum += 1
                    else:
                        destination_island.troops -= troop_unit.troop_count



    def get_next_state(self, player_num):
        """Returns the current environment state representation after the last action."""
        return self.set_init_state(player_num=player_num)

    def update_troop_units(self, player_num, main_surf):
        Variable_Bool = True

        while Variable_Bool:
            for troop_unit in self.troop_units[:]:  # Iterate over a copy to avoid modification issues
                if not troop_unit.has_reached_destination():
                    done = self.is_end_of_game()
                    if done == False:
                        self.running_while_troop_inbound(done, main_surf)

                    dx = troop_unit.destination[0] - (troop_unit.rect.x + troop_unit.rect.width / 2)
                    dy = troop_unit.destination[1] - (troop_unit.rect.y + troop_unit.rect.height / 2)
                    distance = max(1, math.sqrt(dx ** 2 + dy ** 2))
                    dx /= distance
                    dy /= distance
                    troop_unit.rect.x += dx * TROOP_UNIT_SPEED
                    troop_unit.rect.y += dy * TROOP_UNIT_SPEED
                   
                if troop_unit.has_reached_destination():
                    self.resolve_troop_arrival(troop_unit)  # Process conquest logic
                    self.troop_units.remove(troop_unit)  # Remove after arrival
                    Variable_Bool = False # because only 1 troop unit possible to send this works.

            self.troop_units = [troop_unit for troop_unit in self.troop_units if not troop_unit.has_reached_destination()]
            self.troop_units = self.troop_units[:MAX_TROOP_UNITS]
            self.update_FAndE_Troo_Num()


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
            print("ERRRROOOOORRRRRRRRRR!!@!!!!!!!") # Invalid agent type, will never print probably
            return all_possible_actions  

        all_islands = self.get_all_islands()  # All islands (including neutral/enemy)
        

        for source_island in all_my_islands:
            if source_island.troops > 0:
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
                else:
                    print("ERRRROOOOORRRRRRRRRR!!@!!!!!!! 22222")
        return all_possible_actions

    







    def draw_troop_units(self, main_surf):
        for troop_unit in self.troop_units:
            troop_unit.draw(main_surf)


    def draw(self, main_surf):

        # Draw the background image
        main_surf.fill((19,19,19))

        for island in self.islands:
            island.draw(main_surf)  # Draw islands onto main_surf
        self.draw_troop_units(main_surf)  # Draw troop units onto main_surf

    def get_island_index(self, clicked_island):
        # Assuming 'clicked_island' is an Island object, we find its index in the 'self.islands' list.
        try:
            return self.islands.index(clicked_island)
        except ValueError:
            return None  # If the island is not found, return None

            
    def move(self, epoch, main_surf, action_tuple, agent_type, player_num):

        reward = 0
        done = False
        troop_unit_that_was_sent = None
        
        left_clicked_selected_islands = action_tuple[0]
        right_clicked_selected_islands = action_tuple[1]

        attacking_island_index = right_clicked_selected_islands
        attacking_island = self.islands[attacking_island_index] if attacking_island_index is not None else None
        self.selected_friendly_islands = self.islands[left_clicked_selected_islands] if attacking_island_index is not None else None 
        self.selected_enemy_islands = self.islands[left_clicked_selected_islands] if attacking_island_index is not None else None 

        if attacking_island:
            if agent_type == "Human_Agent":
                troop_unit_that_was_sent = self.send_troops(attacking_island,agent_type,player_num)
            elif agent_type == "Random_Agent" or agent_type == "DQN":
                if player_num == "1":
                    total_troops = self.selected_friendly_islands.troops
                    if total_troops > 0:
                        if attacking_island.type in [NEUTRAL, ENEMY]:  
                            defending_troops = attacking_island.troops  

                            # Find the weakest neutral/enemy island
                            weakest_enemy_island = None  
                            weakest_neutral_island = None  
                            least_troops_num = float('inf')  

                            for island in self.islands:  
                                if island.troops < least_troops_num and island.type is not FRIENDLY:  
                                    least_troops_num = island.troops  
                                    if island.type == ENEMY:  
                                        weakest_enemy_island = island  
                                    elif island.type == NEUTRAL:  
                                        weakest_neutral_island = island  

                            # If both an enemy and neutral island have the same lowest troops, prefer the enemy
                            if weakest_enemy_island is not None and weakest_enemy_island.troops == least_troops_num:
                                island_with_least_troops = weakest_enemy_island  
                            else:
                                island_with_least_troops = weakest_neutral_island  

                            # Prioritize attacking enemies over neutrals
                            if weakest_enemy_island is not None and weakest_neutral_island is not None:
                                if attacking_island == island_with_least_troops:  
                                    if attacking_island.type == ENEMY:
                                        reward = 20
                                    if attacking_island.type == NEUTRAL:
                                        reward = -8
                                    if total_troops > defending_troops: 
                                        reward += 5
                                    else:
                                        reward -= 5
                                else:  
                                    reward = -8 
                            else:
                                if attacking_island == island_with_least_troops:  
                                    if total_troops > defending_troops: 
                                        reward = 10
                                else:  
                                    reward = -10  
                        else:  
                            reward = -10  
                        troop_unit_that_was_sent = self.send_troops(attacking_island,agent_type,player_num) 
                    else:
                        print("NOT TRUE total_troops > 0 PLAYER 1")
                elif player_num == "2":
                    total_troops = self.selected_enemy_islands.troops
                    if total_troops > 0:
                        if attacking_island.type in [NEUTRAL, FRIENDLY]:  
                            defending_troops = attacking_island.troops  

                            # Find the weakest neutral/enemy island
                            weakest_friendly_island = None  
                            weakest_neutral_island = None  
                            least_troops_num = float('inf')  

                            for island in self.islands:  
                                if island.troops < least_troops_num and island.type is not ENEMY:  
                                    least_troops_num = island.troops  
                                    if island.type == FRIENDLY:  
                                        weakest_friendly_island = island  
                                    elif island.type == NEUTRAL:  
                                        weakest_neutral_island = island  

                            # If both an enemy and neutral island have the same lowest troops, prefer the enemy
                            if weakest_friendly_island is not None and weakest_friendly_island.troops == least_troops_num:
                                island_with_least_troops = weakest_friendly_island  
                            else:
                                island_with_least_troops = weakest_neutral_island  

                            # Prioritize attacking enemies over neutrals
                            if weakest_friendly_island is not None and weakest_neutral_island is not None:
                                if attacking_island == island_with_least_troops:  
                                    if attacking_island.type == FRIENDLY:
                                        reward = 20
                                    if attacking_island.type == NEUTRAL:
                                        reward = -8
                                    if total_troops > defending_troops: 
                                        reward += 5
                                    else:
                                        reward -= 5 
                                else:  
                                    reward = -8 
                            else:
                                if attacking_island == island_with_least_troops:  
                                    if total_troops > defending_troops: 
                                        reward = 10
                                else:  
                                    reward = -10  
                        else:  
                            reward = -10  # Strong penalty for attacking a friendly island  
                        troop_unit_that_was_sent = self.send_troops(attacking_island,agent_type,player_num)
                    else:
                        print("NOT TRUE total_troops > 0 PLAYER 2")
        
        self.EnemyIslandNum = sum(1 for island in self.islands if island.type == ENEMY)
        self.FriendlyIslandNum = sum(1 for island in self.islands if island.type == FRIENDLY)

        self.update_troop_units(player_num, main_surf)

        done = self.is_end_of_game()
      
        if done:
            player_won =  self.player_won("2")#1 changed because runnign player 2
            if player_won:
                reward += 100
                self.player_1_won += 1
            else:
                reward += -100
                self.player_2_won += 1
        
        return reward, done





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
            return possible_actions


    def get_all_islands(self):
        return self.islands
    
    def get_all_enemy_islands(self):
        return [island for island in self.islands if island.type == ENEMY]
    
    def get_all_friendly_islands(self):
        return [island for island in self.islands if island.type == FRIENDLY]


    def initialize_state_tensor(self):
        return torch.zeros(STATE_SIZE)
