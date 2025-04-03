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
                                waiting_for_input = False  # Exit loop when valid input is received
                                return (self.selected_island_flags, self.right_clicked_island)

            pygame.display.update()



