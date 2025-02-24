import random
from CONSTANTS import *

class Random_Agent:
    def __init__(self):
        self.action = None  # to disable in the future
        self.left_clicked_islands = None#[]
        self.right_clicked_island = None
        self.selected_island_flags = None #[0] * 8  # Assuming 8 islands
        self.send_interval = random.randint(2, 50)  # Set a random interval for troop-sending actions
        self.time_counter = 0  # Track time passed

    def get_Action(self, environment,player_num, events=None, state=None):
        # Increase the time counter every time get_Action is called

        ###### DISABLED FOR DQN LEARNING ########
        # self.time_counter += 1
        # # Only send troops if the time counter has reached the send interval
        # if self.time_counter < self.send_interval:
        #     # Skip sending for this call
        #     print("TIMER RANDOM 1")
        #     return (None,) + (None,)

        # # Reset the time counter and choose a new random interval
        # self.time_counter = 0
        # self.send_interval = random.randint(2, 50)

        # # Random chance to send troops
        # send_troops_chance = 0.5  # 50% chance to send troops
        # if random.random() > send_troops_chance:
        #     print("TIMER RANDOM 2")
        #     return (None,) + (None,)
        ###### DISABLED FOR DQN LEARNING ########

        
        if player_num == "2":
            all_enemy_islands = environment.get_all_enemy_islands()
            #print(f"Enemy islands: {[island.type for island in all_enemy_islands]}")
            all_islands = environment.get_all_islands()
            
            if all_enemy_islands and all_islands:
                # Choose a random enemy island to send troops from
                #source_island_count = random.randint(1, len(all_enemy_islands))
                # print("source_island_count: ", source_island_count)
                action = None
                #for i in range(source_island_count):
                source_island = random.choice(all_enemy_islands)
                source_index = environment.get_island_index(source_island)
                #print(f"Selected source enemy island: Index {source_index}, Type {source_island.type}")

                
                # Choose a random destination island
                destination_island = random.choice(all_islands)
                destination_index = environment.get_island_index(destination_island)
                
                # Create an action to send troops from the source island to the destination island
                
                if source_index is not None:
                    action = source_index  # Set the bit for the chosen source island

                #print("tuple(action) + (destination_index,): ", tuple(action) + (destination_index,))
                # Append the destination island index
                #print("sending troops")
                # if action == 1:
                #     return ((-1,-1))
                # print(f"source_island: {source_island}, source_island.troops: {source_island.troops}")
                return (action,) + (destination_index,) #return ((action,1)) #
                #return (action,) + (destination_index,)
            else:
                # No islands to select
                environment.selected_enemy_islands = None #.clear()
                return ((-1,-1))# return (None,) + (None,)
        elif player_num == "1":
            all_friendly_islands = environment.get_all_friendly_islands()
            #print(f"Enemy islands: {[island.type for island in all_enemy_islands]}")
            all_islands = environment.get_all_islands()
            
            if all_friendly_islands and all_islands:

                #source_island_count = random.randint(1, len(all_friendly_islands))
                # print("source_island_count: ", source_island_count)
                action = None
                #for i in range(source_island_count):

                # Choose a random enemy island to send troops from
                source_island = random.choice(all_friendly_islands)
                source_index = environment.get_island_index(source_island)
                #print(f"Selected source enemy island: Index {source_index}, Type {source_island.type}")

                
                # Choose a random destination island
                destination_island = random.choice(all_islands)
                destination_index = environment.get_island_index(destination_island)
                
                # Create an action to send troops from the source island to the destination island
                if source_index is not None:
                    action = source_index  # Set the bit for the chosen source island

                #print("tuple(action) + (destination_index,): ", tuple(action) + (destination_index,))
                # Append the destination island index
                return (action,) + (destination_index,)
            else:
                # No islands to select
                environment.selected_friendly_islands = None #.clear()
                return ((-1,-1))# return (None,) + (None,)