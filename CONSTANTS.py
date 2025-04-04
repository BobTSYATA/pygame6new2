# CONSTANTS.py:

# RGB Colors
RED = (255, 0, 0)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
LIGHTGRAY = (192, 192, 192)
GREEN = (0, 128, 0)
DARK_GRAY = (50, 50, 50)
ORANGE = (255, 165, 0)

# Island Types
NEUTRAL = 0
FRIENDLY = 1
ENEMY = 2

# FPS
FPS = 60

# Screen Dimensions
WIDTH = 800
HEIGHT = 600
MAIN_SURF_HEIGHT = 500

# Troop Constants
START_TROOPS = 10

TROOP_INCREASE_RATE = 1  # Troops increase per second on conquered islands

# Island Colors
ISLAND_COLORS = {
        "neutral": (0, 0, 204),
        "friendly": (0, 255, 0),
        "enemy": (255, 0, 0)
    }

TROOP_UNIT_TYPES = {
        "neutral": 0,
        "friendly": 1,
        "enemy": -1
    }

BACKGROUND_COLOR = LIGHTGRAY  # Light gray color

ISLAND_RADIUS = 40
ISLAND_DISTANCE_MIN = 100
ISLAND_DISTANCE_MAX = 200

TROOP_UNIT_SPEED = 30#100#
TROOP_UNIT_COLOR = (255, 255, 255)
TROOP_UNIT_RADIUS = 20

MAX_TROOP_UNITS = 2
TROOP_UNIT_DELAY = 0.1

MAX_ISLANDS = 4#8
NUM_VARIABLES = 4

ELEMENTS_PER_TROOP_UNIT = 6
ELEMENTS_PER_ISLAND = 4
STATE_SIZE = (MAX_ISLANDS * ELEMENTS_PER_ISLAND)
