#TroopUnit.py:
import pygame
from CONSTANTS import *
import math
import numpy as np
import cv2

class TroopUnit(pygame.sprite.Sprite):
    neon_effects = {}

    def __init__(self, pos, color, troop_count, destination) -> None:
        super().__init__()
        pygame.font.init()  # Ensure fonts are initialized
        self.font = pygame.font.SysFont("arial", 12)  # Cache font for troop count
        self.color = color
        self.type = TROOP_UNIT_TYPES.get(
            "enemy" if color == ISLAND_COLORS["enemy"] else
            "friendly" if color == ISLAND_COLORS["friendly"] else
            "neutral"
        )
        self.troop_count = troop_count
        self.destination = destination  # Store the destination coordinates
        self.image = pygame.Surface((2 * TROOP_UNIT_RADIUS, 2 * TROOP_UNIT_RADIUS))
        self.rect = self.image.get_rect(midbottom=pos)
        self.mask = pygame.mask.from_surface(self.image)

        self.draw_troop_count()

    @staticmethod
    def create_neon_circle(radius, color):
        # Check if the neon effect is already cached for this radius and color
        if (radius, color) in TroopUnit.neon_effects:
            return TroopUnit.neon_effects[(radius, color)]
        
        # Create a surface for the neon glow with padding for the blur effect
        surf_size = radius * 10
        surf = pygame.Surface((surf_size, surf_size), pygame.SRCALPHA)
        
        # Draw the circle at the center of the surface
        pygame.draw.circle(surf, color, (surf_size // 2, surf_size // 2), radius)
        
        # Convert the surface to a format OpenCV can process
        surf_alpha = surf.convert_alpha()
        rgb = pygame.surfarray.array3d(surf_alpha)
        alpha = pygame.surfarray.array_alpha(surf_alpha).reshape((*rgb.shape[:2], 1))
        image = np.concatenate((rgb, alpha), 2)
        
        # Apply Gaussian and regular blur for a subtle neon glow effect
        blurred_image = cv2.GaussianBlur(image, (7, 7), sigmaX=5, sigmaY=5)
        blurred_image = cv2.blur(blurred_image, (20, 20))
        
        # Convert back to a Pygame surface and cache it
        neon_surf = pygame.image.frombuffer(blurred_image.flatten(), blurred_image.shape[1::-1], 'RGBA')
        TroopUnit.neon_effects[(radius, color)] = neon_surf
        return neon_surf

    def draw_troop_count(self):
        text_surface = self.font.render(str(self.troop_count), True, WHITE)
        text_rect = text_surface.get_rect(center=(TROOP_UNIT_RADIUS, TROOP_UNIT_RADIUS))
        self.image.blit(text_surface, text_rect)

    def has_reached_destination(self):
        # Calculate the distance between the troop unit's current position and the destination
        dx = self.destination[0] - self.rect.centerx
        dy = self.destination[1] - self.rect.centery
        distance = math.sqrt(dx ** 2 + dy ** 2)
        # Check if the distance is less than a threshold (e.g., TROOP_UNIT_SPEED)
        return distance < TROOP_UNIT_SPEED

    def draw(self, surface):
         # Draw cached neon glow
        neon_effect = self.create_neon_circle(TROOP_UNIT_RADIUS, self.color)
        neon_rect = neon_effect.get_rect(center=self.rect.center)
        surface.blit(neon_effect, neon_rect, special_flags=pygame.BLEND_PREMULTIPLIED)

        pygame.draw.circle(surface, self.color, self.rect.center, TROOP_UNIT_RADIUS)
        # Render troop count text
        font = pygame.font.SysFont("arial", 12)
        text_surface = font.render(str(self.troop_count), True, WHITE)
        text_rect = text_surface.get_rect(center=self.rect.center)
        surface.blit(text_surface, text_rect)