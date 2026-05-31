import os


MODEL6_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.dirname(MODEL6_DIR)
PROJECT_ROOT = os.path.dirname(NETWORK_DIR)

DATASET_CSV = os.path.join(PROJECT_ROOT, "dataset", "relative_coordinates_export.csv")

DEFAULT_TD3_MODEL = os.path.join(NETWORK_DIR, "Model1", "checkpoints", "td3_resume_latest.zip")
FALLBACK_TD3_MODEL = os.path.join(NETWORK_DIR, "Model1", "td3_airsim_uav_model.zip")

# Raycast-based occupancy detection (top-down).
# NED coordinates: z more negative = higher, z more positive = lower.
OCCUPANCY_GROUND_Z = 0.0           # flat AirSim world ground z in NED coordinates
OCCUPANCY_RAY_ABOVE_GROUND = 60.0  # vertical ray starts this many meters above ground
OCCUPANCY_GROUND_CLEARANCE = 1.0   # ray ends this many meters above ground, avoiding ground collision
OCCUPANCY_RAY_PROGRESS_ROWS = 20   # print progress every N grid rows while building occupancy

# Upper planner: default is online occupancy-grid A* from AirSim scene obstacles.
UPPER_PLANNER = "occupancy"  # one of: "occupancy", "csv"
OCCUPANCY_RESOLUTION = 1.0
OCCUPANCY_BOUNDS_MARGIN = 60.0
OCCUPANCY_MIN_PLAN_SPAN = 140.0
OCCUPANCY_MIN_X = -140.0
OCCUPANCY_MAX_X = 170.0
OCCUPANCY_MIN_Y = -200.0
OCCUPANCY_MAX_Y = 190.0
OCCUPANCY_OBSTACLE_RADIUS = 2.0
OCCUPANCY_SAFETY_MARGIN = 1.5
OCCUPANCY_NEAREST_FREE_RADIUS = 12.0
OCCUPANCY_START_GOAL_CLEAR_RADIUS = 0.0
OCCUPANCY_REQUIRE_NONEMPTY_MAP = True
OCCUPANCY_FALLBACK_LARGE_OBJECT_MIN_SIDE = 6.0
OCCUPANCY_ALLOW_DIAGONAL = True
OCCUPANCY_RADIUS_FALLBACKS = (
    (2.0, 1.5),
    (1.5, 1.0),
    (1.0, 0.6),
    (0.6, 0.3),
)
LOCAL_TARGET_SPACING = 40
LOCAL_TARGET_MIN_SPACING = 20
LOCAL_TARGET_KEEP_TURNS = True

# Fallback planner: graph A* over offline feasible navigation points.
GRAPH_K_NEIGHBORS = 8
GRAPH_MAX_EDGE_DISTANCE = 35.0
GRAPH_MAX_Z_DIFF = 8.0
GRAPH_VERTICAL_COST_WEIGHT = 0.35
GRAPH_REGION_CHANGE_COST = 2.0
GRAPH_FALLBACKS = (
    (8, 35.0),
    (12, 50.0),
    (18, 75.0),
    (30, 120.0),
)

# Lower TD3 execution.
SEGMENT_MAX_STEPS = 180
SEGMENT_GOAL_TOLERANCE = 2.0
INTERMEDIATE_AXIS_TOLERANCE = 3.0
FINAL_AXIS_TOLERANCE = 1.5
STOP_ON_SEGMENT_FAILURE = True
DETERMINISTIC_POLICY = True

# AirSim waypoint safety validation for upper graph nodes.
VALIDATE_WAYPOINTS_WITH_AIRSIM = False
OBSTACLE_SEGMENTATION_ID = 2
WAYPOINT_OBSTACLE_CLEARANCE = 3.0
WAYPOINT_SURROUNDED_RADIUS = 8.0
WAYPOINT_SURROUNDED_MIN_FREE_DIRECTIONS = 3
WAYPOINT_SURROUNDED_RAYS = 8
WAYPOINT_RAY_CLEARANCE_DIST = 4.0  # simCastRay must find >= this distance in all 8 directions
OBSTACLE_OBJECT_PATTERNS = (
    "building",
    "wall",
    "house",
    "roof",
    "block",
    "ad_column",
    "adstand",
    "ad_stand",
    "air_conditioner",
    "awning",
    "bench",
    "bicycle",
    "bike_stand",
    "bus_stop",
    "chair",
    "flower",
    "flower_pot",
    "food_cart",
    "food_stand",
    "hydrant",
    "kiosk",
    "lamp",
    "plant",
    "plant_bush",
    "railing",
    "scaffolding",
    "sign",
    "street_barier",
    "street_clock",
    "street_lamp",
    "street_pole",
    "street_sign",
    "sunshade",
    "table",
    "traffic_lights",
    "trashcan",
    "tree_stand",
)

# Visualization.
VISUALIZE_GLOBAL_PATH = True
VISUAL_Z_OFFSET = 20.0
