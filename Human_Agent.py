import pygame
from CONSTANTS import *

class Human_Agent:
    def __init__(self):
        self.left_clicked_islands = None
        self.right_clicked_island = None
        self.selected_island_flags = None

    def get_Action(self, environment, player_num, events=None, state=None):
        if events is None:
            events = []

        print(f"Waiting for Player {player_num} to make a move...")
        self.left_clicked_islands = None  # Reset previous selection
        self.selected_island_flags = None
        self.right_clicked_island = None

        waiting_for_input = True

        while waiting_for_input:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    exit()

                if event.type == pygame.MOUSEBUTTONDOWN:
                    clicked_island = environment.get_clicked_island()
                    
                    if clicked_island:
                        if pygame.mouse.get_pressed()[0]:  # Left click
                            if clicked_island.type == FRIENDLY:
                                # print(f"Selected friendly island: {clicked_island}")
                                self.left_clicked_islands = clicked_island
                                self.selected_island_flags = environment.get_island_index(clicked_island)
                            else:
                                print("Invalid selection: Not a friendly island.")
                                self.left_clicked_islands = None
                                self.selected_island_flags = None

                        elif pygame.mouse.get_pressed()[2]:  # Right click
                            if self.left_clicked_islands is None:
                                print("No valid island selected. Cannot attack.")
                                continue  

                            if self.left_clicked_islands.type != FRIENDLY:
                                print("Previously selected island is no longer yours! Re-select a valid island.")
                                self.left_clicked_islands = None
                                self.selected_island_flags = None
                                continue

                            self.right_clicked_island = environment.get_island_index(clicked_island)

                            if self.right_clicked_island is not None:
                                # print(f"Attacking island {self.right_clicked_island} from {self.selected_island_flags}")
                                waiting_for_input = False  # Exit loop when valid input is received
                                return (self.selected_island_flags, self.right_clicked_island)

            pygame.display.update()







    # def get_Action(self, environment, player_num,events=None, state=None):
    #     if player_num == "1":
    #         for event in events:
    #             if event.type == pygame.MOUSEBUTTONDOWN:
    #                 if pygame.mouse.get_pressed()[0]:  # Left click
    #                     clicked_island = environment.get_clicked_island()
                        
    #                     if state is not None and environment.legal("LEFT_CLICK", 1,):
    #                         if clicked_island:
    #                             if self.indexCustom == 1:
    #                                 self.selected_island_flags = None#[0] * 8
    #                                 self.right_clicked_island = None
    #                                 self.indexCustom = 0
    #                             if clicked_island.type == FRIENDLY:
    #                                 if clicked_island != None:
    #                                     self.left_clicked_islands = clicked_island
    #                                     #print(f"Left clicked island at ({clicked_island.x}, {clicked_island.y})")

    #                                 # Mark the island as selected in the flags
    #                                 island_index = environment.get_island_index(clicked_island)
    #                                 if island_index is not None:
    #                                     self.selected_island_flags = island_index  # Mark island as selected
    #                                     #print("selected_island_flags: ", self.selected_island_flags)
                                
    #                             # Retrieve and print possible actions
    #                             # possible_actions = environment.get_possible_actions("LEFT_CLICK",1)

    #                         else:
    #                             # Clear selected islands if clicked elsewhere
    #                             self.left_clicked_islands = None
    #                             environment.selected_friendly_islands = None
    #                             self.selected_island_flags = None

    #                 elif pygame.mouse.get_pressed()[2]:  # Right click
    #                     clicked_island = environment.get_clicked_island()

    #                     if state is not None and environment.legal("RIGHT_CLICK", 1):
    #                         if clicked_island:
    #                             self.right_clicked_island = environment.get_island_index(clicked_island)
    #                             #print("tuple(self.selected_island_flags) + (self.right_clicked_island,): ", tuple(self.selected_island_flags) + (self.right_clicked_island,))
    #                             self.indexCustom = 1
    #                             return (self.selected_island_flags,) + (self.right_clicked_island,)

    #         # Return island flags and None if no right-clicked island

    #         #PlaceHolder = [0] * 8

    #         # return (None,) + (None,)
    #         return ((-1,-1))