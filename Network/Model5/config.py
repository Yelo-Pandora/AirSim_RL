import os


MODEL5_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(MODEL5_DIR))

# AirSim / vehicle
VEHICLE_NAME = "Drone1"
CRUISE_ALTITUDE_Z = -5.0
SAFE_ALTITUDE_Z = -28.0  # z axis climb altitude for obstacle clearance (NED: more negative = higher)
MAX_SPEED = 2.0
WAYPOINT_REACHED_DIST = 1.0
CONTROL_DT = 0.1
WAYPOINT_TIMEOUT_SEC = 8.0

# A* grid.  AirSim uses NED: x/y horizontal, z down.
MAP_RESOLUTION = 1.0
MAP_X_MIN = -80.0
MAP_X_MAX = 80.0
MAP_Y_MIN = -110.0
MAP_Y_MAX = 130.0
OBSTACLE_RADIUS = 2.5
SAFETY_MARGIN = 1.5
PLANNING_RADIUS_SCHEDULE = (
    (2.5, 1.5),
    (1.5, 1.0),
    (1.0, 0.5),
    (0.5, 0.2),
)
NEAREST_FREE_SEARCH_RADIUS = 12.0
ALLOW_DIAGONAL = True

# Dataset endpoints used by Model3/Model4 style experiments.
DATASET_CSV = os.path.join(PROJECT_ROOT, "dataset", "relative_coordinates_export.csv")

# Segmentation convention inherited from Model2:
# 0 unknown/background, 1 traversable road-like surface, 2 obstacle.
FREE_SEGMENTATION_IDS = {1}
BLOCKED_SEGMENTATION_IDS = {2}

FREE_OBJECT_PATTERNS = (
    "sidewalk",
    "street",
    "curb",
)

OBSTACLE_OBJECT_PATTERNS = (
    "building",
    "ad_column",
    "adstand",
    "ad_stand",
    "air_conditioner",
    "awning",
    "bench",
    "bicycle",
    "bike_stand",
    "blanket",
    "bus_stop",
    "chair",
    "cusion",
    "cutlery",
    "dec_glass",
    "drainage",
    "drape",
    "flower",
    "flower_pot",
    "food_cart",
    "food_stand",
    "fork",
    "frame_menu",
    "glass",
    "hydrant",
    "infobox",
    "kiosk",
    "knife",
    "lamp",
    "lamp_facade",
    "leafdry",
    "logo",
    "mailbox",
    "manhole",
    "menu",
    "metro_entrance",
    "napkin",
    "newspaper",
    "parking_meter",
    "pepper",
    "plant",
    "plant_bush",
    "plate",
    "pot",
    "railing",
    "scaffolding",
    "sign",
    "singpost",
    "spice_maker",
    "street_barier",
    "street_clock",
    "street_lamp",
    "street_pole",
    "street_sign",
    "sunshade",
    "table",
    "tablecloth",
    "table_set",
    "thyme",
    "ticket_machine",
    "traffic_lights",
    "transformator",
    "trashcan",
    "tree_stand",
    "wheel_stopper",
)
