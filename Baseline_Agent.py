import random
from CONSTANTS import *

class Baseline_Agent:
    def __init__(self):
        self.action = None  # to disable in the future
        self.left_clicked_islands = None  # []
        self.right_clicked_island = None
        self.selected_island_flags = None  # [0] * 8  # Assuming 8 islands
        self.send_interval = random.randint(2, 50)  # Set a random interval for troop-sending actions
        self.time_counter = 0  # Track time passed

    def get_Action(self, environment, player_num, events=None, state=None):
        if player_num == "2":
            all_enemy_islands = environment.get_all_enemy_islands()
            all_islands = environment.get_all_islands()

            if all_enemy_islands and all_islands:
                action = None

                # Decide behavior: 20% chance to attack only player's islands, 50% random
                if random.random() < 0.2:  # Attack only player's conquered islands
                    target_islands = environment.get_all_friendly_islands()  # Opponent's islands
                    if not target_islands:
                        # return (None,) + (None,)
                        return ((-1,-1))
                        

                    destination_island = random.choice(target_islands)
                else:  # Random attack
                    destination_island = random.choice(all_islands)

                # Choose a random enemy island to send troops from
                source_island = random.choice(all_enemy_islands)
                source_index = environment.get_island_index(source_island)
                destination_index = environment.get_island_index(destination_island)

                if source_index is not None:
                    action = source_index  # Set the bit for the chosen source island

                return (action,) + (destination_index,)

            else:
                environment.selected_enemy_islands = None
                # return (None,) + (None,)
                return ((-1,-1))

        elif player_num == "1":
            all_friendly_islands = environment.get_all_friendly_islands()
            all_islands = environment.get_all_islands()

            if all_friendly_islands and all_islands:
                action = None

                # Decide behavior: 50% chance to attack only player's islands, 50% random
                if random.random() < 0.5:  # Attack only player's conquered islands
                    target_islands = environment.get_all_enemy_islands()  # Opponent's islands
                    if not target_islands:
                        # return (None,) + (None,)
                        return ((-1,-1))

                    destination_island = random.choice(target_islands)
                else:  # Random attack
                    destination_island = random.choice(all_islands)

                # Choose a random friendly island to send troops from
                source_island = random.choice(all_friendly_islands)
                source_index = environment.get_island_index(source_island)
                destination_index = environment.get_island_index(destination_island)

                if source_index is not None:
                    action = source_index  # Set the bit for the chosen source island

                return (action,) + (destination_index,)

            else:
                environment.selected_friendly_islands = None
                # return (None,) + (None,)
                return ((-1,-1))



        # if player_num == "2":
                #     all_enemy_islands = environment.get_all_enemy_islands()
                #     all_islands = environment.get_all_islands()

                #     if all_enemy_islands and all_islands:
                #         action = None

                #         # # Decide behavior: 50% chance to attack only player's islands, 50% random
                #         # if random.random() < 0.5:  # Attack only player's conquered islands
                #         #     target_islands = environment.get_all_friendly_islands()  # Opponent's islands
                #         #     if not target_islands:
                #         #         return (None,) + (None,)

                #         #     destination_island = random.choice(target_islands)
                #         # else:  # Random attack
                #         #     destination_island = random.choice(all_islands)

                #         # # Choose a random enemy island to send troops from
                #         source_island =all_enemy_islands[0]
                #         #print("source_island: ", source_island)
                #         source_index = environment.get_island_index(source_island)
                #         #print("source_index: ", source_index)
                #         # destination_index = environment.get_island_index(destination_island)

                #         if source_index is not None:
                #             action = source_index  # Set the bit for the chosen source island
                #         #print("sfgsdfgdfg")
                #         return (action,) + (1,)

                #     else:
                #         environment.selected_enemy_islands = None
                #         return (None,) + (None,)