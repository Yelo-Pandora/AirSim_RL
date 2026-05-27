import os


MODEL6_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.dirname(MODEL6_DIR)
PROJECT_ROOT = os.path.dirname(NETWORK_DIR)

DATASET_CSV = os.path.join(PROJECT_ROOT, "dataset", "relative_coordinates_export.csv")

DEFAULT_TD3_MODEL = os.path.join(NETWORK_DIR, "Model1", "checkpoints", "td3_resume_latest.zip")
FALLBACK_TD3_MODEL = os.path.join(NETWORK_DIR, "Model1", "td3_airsim_uav_model.zip")

# Upper planner: graph A* over feasible navigation points.
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
INTERMEDIATE_AXIS_TOLERANCE = 0.8
FINAL_AXIS_TOLERANCE = 1.5
STOP_ON_SEGMENT_FAILURE = True
DETERMINISTIC_POLICY = True

# AirSim waypoint safety validation for upper graph nodes.
VALIDATE_WAYPOINTS_WITH_AIRSIM = True
WAYPOINT_OBSTACLE_CLEARANCE = 3.0
WAYPOINT_SURROUNDED_RADIUS = 8.0
WAYPOINT_SURROUNDED_MIN_FREE_DIRECTIONS = 3
WAYPOINT_SURROUNDED_RAYS = 8
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
