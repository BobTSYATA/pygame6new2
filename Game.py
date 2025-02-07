import pygame
from Environment import Environment
from Human_Agent import Human_Agent
from Random_Agent import Random_Agent
from CONSTANTS import *
import math
import numpy as np
import cv2
from DQN_Agent import DQN_Agent  # Import the DQN_Agent class

# Define the initial screen
def start_screen():
    pygame.init()
    win = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption('Select Agent Types')

    # Colors and fonts
    colors = {
        'button': (0,0,139),
        'highlight': (0,0,205),  # bright yellow highlight
        'text': (255, 255, 255),  # white text
        'neon': (0,0,255),  # neon green for glowing effects
    }
    font = pygame.font.SysFont('Arial', 48)
    small_font = pygame.font.SysFont('Arial', 36)
    clock = pygame.time.Clock()

    # Default selections
    player1_agent = 'Random'
    player2_agent = 'Random'
    gradient_offset = 0  # Starting position for the gradient

    def brighten_color(color, factor=1.5):
        return tuple(min(255, int(c / factor)) for c in color)

    def create_neon_glow_effect(width, height, color, border_radius=15):
        # Create a surface larger than the button to accommodate the glow
        surf_size = (width + 60, height + 60)  # Increased size for the glow effect
        surf = pygame.Surface(surf_size, pygame.SRCALPHA)

        # Draw a rounded rectangle for the glow
        button_rect = pygame.Rect(30, 30, width, height)
        pygame.draw.rect(surf, color, button_rect, border_radius=border_radius)

        # Convert to a format that OpenCV can use
        surf_alpha = surf.convert_alpha()
        rgb = pygame.surfarray.array3d(surf_alpha)
        alpha = pygame.surfarray.array_alpha(surf_alpha).reshape((*rgb.shape[:2], 1))
        image = np.concatenate((rgb, alpha), 2)

        # Apply Gaussian blur for a glowing effect
        blur_kernel_size = (21, 21)
        blur_sigma = 20
        image = cv2.GaussianBlur(image, ksize=blur_kernel_size, sigmaX=blur_sigma, sigmaY=blur_sigma)

        # Rotate the image by 90 degrees
        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        # Convert back to a pygame surface
        neon_surf = pygame.image.frombuffer(image.flatten(), image.shape[1::-1], 'RGBA')
        return neon_surf

    def draw_moving_gradient(surface, offset):
        # Only a small range close to black
        base_shade = 5  # Base color (almost black)
        shade_range = 5  # Only a slight variation in brightness

        for i in range(HEIGHT):
            adjusted_i = (i + offset) % HEIGHT  # Adjust line position with offset
            shade = base_shade + int((adjusted_i / HEIGHT) * shade_range)
            pygame.draw.line(surface, (shade, shade, shade), (0, i), (WIDTH, i))


    while True:

        # Draw animated gradient background
        draw_moving_gradient(win, gradient_offset)
        gradient_offset -= 1  # Move the gradient down by 1 pixel per frame
        if gradient_offset >= HEIGHT:  # Reset the offset when it reaches the bottom
            gradient_offset = 0


        # Title text with shadow effect
        title_text = font.render("Select Agent Types", True, (255, 255, 255))
        win.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2 + 2, 102))
        win.blit(title_text, (WIDTH // 2 - title_text.get_width() // 2, 100))

        # Player 1 buttons with hover effects
        mouse_pos = pygame.mouse.get_pos()

        # Usage within your main loop
        def draw_button_with_glow(x, y, width, height, color, text, font, highlight=False):
            if highlight:
                # Calculate a brighter color for the glow effect
                brightened_color = brighten_color(color)
                neon_effect = create_neon_glow_effect(width, height, brightened_color)
                neon_rect = neon_effect.get_rect(center=(x + width // 2, y + height // 2))
                win.blit(neon_effect, neon_rect, special_flags=pygame.BLEND_PREMULTIPLIED)
            # Draw the actual button
            pygame.draw.rect(win, color, (x, y, width, height), border_radius=15)
            text_surface = font.render(text, True, colors['text'])
            win.blit(text_surface, (x + (width - text_surface.get_width()) // 2, y + (height - text_surface.get_height()) // 2))


        # Define the offset to move buttons up
        offset_y = 50  # Move buttons 50 pixels up

        # Check if mouse is over buttons for hover effect
        player1_hover_human = 50 < mouse_pos[0] < 320 and 300 - offset_y < mouse_pos[1] < 350 - offset_y
        player1_hover_random = 50 < mouse_pos[0] < 320 and 370 - offset_y < mouse_pos[1] < 420 - offset_y
        player1_hover_dqn = 50 < mouse_pos[0] < 320 and 440 - offset_y < mouse_pos[1] < 490 - offset_y

        # Draw buttons with neon effect when hovered for Player 1 (vertical layout on the left)
        draw_button_with_glow(50, 300 - offset_y, 270, 50, colors['highlight'] if player1_agent == 'Human' else colors['button'], 'Player 1: Human', small_font, player1_hover_human)
        draw_button_with_glow(50, 370 - offset_y, 270, 50, colors['highlight'] if player1_agent == 'Random' else colors['button'], 'Player 1: Random', small_font, player1_hover_random)
        draw_button_with_glow(50, 440 - offset_y, 270, 50, colors['highlight'] if player1_agent == 'DQN' else colors['button'], 'Player 1: DQN', small_font, player1_hover_dqn)

        # Player 2 buttons with hover effects (vertical layout on the right)
        # Align Player 2 buttons with Player 1's first and second buttons at y = 300 - offset_y and y = 370 - offset_y
        player2_hover_random = 470 < mouse_pos[0] < 740 and 300 - offset_y < mouse_pos[1] < 350 - offset_y
        player2_hover_dqn = 470 < mouse_pos[0] < 740 and 370 - offset_y < mouse_pos[1] < 420 - offset_y

        # Draw buttons with neon effect when hovered for Player 2
        draw_button_with_glow(470, 300 - offset_y, 270, 50, colors['highlight'] if player2_agent == 'Random' else colors['button'], 'Player 2: Random', small_font, player2_hover_random)
        draw_button_with_glow(470, 370 - offset_y, 270, 50, colors['highlight'] if player2_agent == 'DQN' else colors['button'], 'Player 2: DQN', small_font, player2_hover_dqn)

        # Play button with neon effect
        play_button_hover = 300 < mouse_pos[0] < 500 and 500 < mouse_pos[1] < 560
        NEW_GREEN = (124, 252, 0)
        draw_button_with_glow(300, 500, 200, 60, NEW_GREEN, "Play", font, play_button_hover)

        pygame.display.flip()

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None, None
            if event.type == pygame.MOUSEBUTTONDOWN:
                pos = pygame.mouse.get_pos()
                # Check if Play button is clicked
                if 300 < pos[0] < 500 and 500 < pos[1] < 560:
                    pygame.quit()
                    return player1_agent, player2_agent
                # Check Player 1 selections
                if 50 < pos[0] < 320 and 300 - offset_y < pos[1] < 350 - offset_y:
                    player1_agent = 'Human'
                if 50 < pos[0] < 320 and 370 - offset_y < pos[1] < 420 - offset_y:
                    player1_agent = 'Random'
                if 50 < pos[0] < 320 and 440 - offset_y < pos[1] < 490 - offset_y:
                    player1_agent = 'DQN'

                # Check Player 2 selections
                if 470 < pos[0] < 740 and 300 - offset_y < pos[1] < 350 - offset_y:
                    player2_agent = 'Random'
                if 470 < pos[0] < 740 and 370 - offset_y < pos[1] < 420 - offset_y:
                    player2_agent = 'DQN'
        clock.tick(60)

def main(player1_type, player2_type):
    pygame.init()
    clock = pygame.time.Clock()
    environment = Environment()
    main_surf = pygame.Surface((WIDTH, HEIGHT - 100))
    main_surf.fill(LIGHTGRAY)

    # print("player1_type: ", player1_type, " player2_type: ", player2_type)

    # Initialize agents based on selected types
    if player1_type == 'Human':
        player_1 = Human_Agent()
    elif player1_type == 'DQN':
        player_1 = DQN_Agent()
    elif player1_type == 'Random':
        player_1 = Random_Agent()
    
    if player2_type == 'Human':
        player_2 = Human_Agent()
    elif player2_type == 'DQN':
        player_2 = DQN_Agent()
    elif player2_type == 'Random':
        player_2 = Random_Agent()
    run = True
    while run:
        events = pygame.event.get()
        for event in events:
            if event.type == pygame.QUIT:
                run = False

        clock.tick()
        #print(clock.get_fps())

        # Get the current state from the environment
        state = environment.state()
 
        # Player 1's action
        action_tuple = player_1.get_Action(environment, player_num="1", events=events, state=state)
        agent_type1 = "Random_Agent"
        agent_type2 = "Random_Agent"
        if player1_type == 'Human':
            agent_type1 = "Human_Agent"
        elif player1_type == "DQN":
            agent_type1 = "DQN"
        elif player1_type == "Random":
            agent_type1 = "Random_Agent"

        if player2_type == 'Human':
            agent_type2 = "Human_Agent"
        elif player2_type == "DQN":
            agent_type2 = "DQN"
        elif player2_type == "Random":
            agent_type2 = "Random_Agent"
        player_done = environment.move(
            action_tuple,
            agent_type=agent_type1,
            player_num="1",
            state=state
        )
        environment.draw_header(player_done, main_surf)
        
        #environment.all_possible_actions_for_agent(1)

        # Player 2's action
        action_tuple_random = player_2.get_Action(environment, player_num="2", events=events, state=state)
        player_done_random = environment.move(
            action_tuple_random,
            agent_type=agent_type2,
            player_num="2",
            state=state
        )
        environment.draw_header(player_done_random, main_surf)
        
        if environment.should_close_game:
            break  # Exit the game loop

if __name__ == "__main__":
    player1_type, player2_type = start_screen()
    if player1_type and player2_type:
        main(player1_type, player2_type)