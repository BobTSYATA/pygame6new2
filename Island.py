import pygame
from CONSTANTS import *
from TroopUnit import TroopUnit
import math
import cv2
import numpy as np

class Island(pygame.sprite.Sprite):
    neon_effects = {}  # Dictionary to cache neon effects by (radius, color) key

    def __init__(self, x, y, radius, color, island_type, troop_num = START_TROOPS) -> None:
        super().__init__()
        pygame.font.init()  # Ensure fonts are initialized
        self.font = pygame.font.SysFont("arial", 16)  # Cache font for troop count
        self.x = x
        self.y = y
        self.radius = radius
        self.color = color
        self.type = island_type
        self.troops = troop_num

        # Initialize rect based on position and radius (make sure it's centered)
        self.rect = pygame.Rect(self.x - self.radius, self.y - self.radius, self.radius * 2, self.radius * 2)

    @staticmethod # this just means that it doesn't get self instance
    def create_neon_circle(radius, color):
        # Check if the neon effect is already cached
        if (radius, color) in Island.neon_effects:
            return Island.neon_effects[(radius, color)]
        
        # Create a surface for the neon glow with padding for the blur effect
        surf_size = radius * 10
        surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        
        # Draw the circle at the center of the surface
        pygame.draw.circle(surf, color, (surf_size // 2, surf_size // 2), radius)
        
        # Convert to OpenCV-compatible format
        surf_alpha = surf.convert_alpha()
        rgb = pygame.surfarray.array3d(surf_alpha)
        
        alpha = pygame.surfarray.array_alpha(surf_alpha).reshape((*rgb.shape[:2], 1))
        image = np.concatenate((rgb, alpha), 2)
        
        # Apply a strong Gaussian and blur effect for the neon glow
        cv2.GaussianBlur(image, (21, 21), sigmaX=15, sigmaY=15, dst=image)
        cv2.blur(image, (60, 60), dst=image)
        
        # Convert back to a Pygame surface and cache it
        neon_surf = pygame.image.frombuffer(image.flatten(), image.shape[1::-1], 'RGBA')
        Island.neon_effects[(radius, color)] = neon_surf
        return neon_surf
    

    def draw(self, screen):
        # Draw cached neon glow
        neon_effect = self.create_neon_circle(self.radius, self.color)
        neon_rect = neon_effect.get_rect(center=self.rect.center)
        screen.blit(neon_effect, neon_rect, special_flags=pygame.BLEND_PREMULTIPLIED)

        # Draw the main circle (island)
        pygame.draw.circle(screen, self.color, (self.x, self.y), self.radius)
        
        # Render troop count text
        text_surface = self.font.render(str(self.troops), True, WHITE)
        text_rect = text_surface.get_rect(center=(self.x, self.y))
        screen.blit(text_surface, text_rect)  # Draw text on the island