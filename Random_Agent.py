import random
from CONSTANTS import *

class Random_Agent:
    def __init__(self):
        self.action = None 
        self.left_clicked_islands = None
        self.right_clicked_island = None
        self.selected_island_flags = None 
        self.send_interval = random.randint(2, 50)  # Set a random interval for troop-sending actions
        self.time_counter = 0  # Track time passed

    def get_Action(self, environment,player_num, events=None, state=None):

        if player_num == "2":
            all_enemy_islands = environment.get_all_enemy_islands()
            all_islands = environment.get_all_islands()
            
            if all_enemy_islands and all_islands:
                # Choose a random enemy island to send troops from
                action = None
                source_island = random.choice(all_enemy_islands)
                source_index = environment.get_island_index(source_island)

                # Choose a random destination island
                destination_island = random.choice(all_islands)
                destination_index = environment.get_island_index(destination_island)
                

                if source_index is not None:
                    action = source_index  # Set the bit for the chosen source island


                return (action,) + (destination_index,) 
            else:
                # No islands to select
                environment.selected_enemy_islands = None 
                return ((-1,-1))
        elif player_num == "1":
            all_friendly_islands = environment.get_all_friendly_islands()
            all_islands = environment.get_all_islands()
            
            if all_friendly_islands and all_islands:
                action = None

                # Choose a random enemy island to send troops from
                source_island = random.choice(all_friendly_islands)
                source_index = environment.get_island_index(source_island)

                # Choose a random destination island
                destination_island = random.choice(all_islands)
                destination_index = environment.get_island_index(destination_island)
                
                # Create an action to send troops from the source island to the destination island
                if source_index is not None:
                    action = source_index  # Set the bit for the chosen source island

                # Append the destination island index
                return (action,) + (destination_index,)
            else:
                # No islands to select
                environment.selected_friendly_islands = None
                return ((-1,-1))