import os


MODEL6_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.dirname(MODEL6_DIR)
PROJECT_ROOT = os.path.dirname(NETWORK_DIR)

DATASET_CSV = os.path.join(PROJECT_ROOT, "dataset", "relative_coordinates_export.csv")

DEFAULT_TD3_MODEL = os.path.join(NETWORK_DIR, "TD3_base", "checkpoints", "td3_resume_latest.zip")
FALLBACK_TD3_MODEL = os.path.join(NETWORK_DIR, "TD3_base", "td3_airsim_uav_model.zip")

# Raycast-based occupancy detection (top-down).
# NED coordinates: z more negative = higher, z more positive = lower.
OCCUPANCY_GROUND_Z = 0.0           # flat AirSim world ground z in NED coordinates
OCCUPANCY_RAY_ABOVE_GROUND = 60.0  # vertical ray starts this many meters above ground
OCCUPANCY_GROUND_CLEARANCE = 1.0   # ray ends this many meters above ground, avoiding ground collision
OCCUPANCY_RAY_PROGRESS_ROWS = 20   # print progress every N grid rows while building occupancy

# Upper planner: default is online occupancy-grid A* from AirSim scene obstacles.
UPPER_PLANNER = "occupancy"  # one of: "occupancy", "csv"
OCCUPANCY_USE_OBJECT_BOUNDS = True
OCCUPANCY_USE_LOS_FALLBACK = False
OCCUPANCY_INCLUDE_SEGMENTATION_OBSTACLES = False
OCCUPANCY_USE_TOPDOWN_MAP = True
OCCUPANCY_TOPDOWN_PREINFLATED = True
OCCUPANCY_TOPDOWN_METADATA = os.path.join(
    MODEL6_DIR,
    "topdown_depth",
    "topdown_depth_20260601_165808_occupancy_h10_infl2.json",
)
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
OCCUPANCY_ASTAR_MIN_CLEARANCE = 5.0
OCCUPANCY_ASTAR_PREFERRED_CLEARANCE = 8.0
OCCUPANCY_ASTAR_CLEARANCE_COST_WEIGHT = 8.0
OCCUPANCY_LEGACY_FALLBACK = True
OCCUPANCY_RADIUS_FALLBACKS = (
    (2.0, 1.5),
    (1.5, 1.0),
    (1.0, 0.6),
    (0.6, 0.3),
)
LOCAL_TARGET_SPACING = 20
LOCAL_TARGET_MIN_SPACING = 10
LOCAL_TARGET_KEEP_TURNS = True
LOCAL_TARGET_CLEARANCE = 6.0
LOCAL_TARGET_CORRIDOR_CLEARANCE = 0.0
LOCAL_TARGET_CORRIDOR_SAMPLE_STEP = 1.0
LOCAL_TARGET_SAFETY_PROFILES = (
    # Values include any pre-inflation already baked into the top-down map.
    # Keep waypoints away from walls; keep corridor as LOS-only for feasibility.
    (6.0, 0.0),
    (5.0, 0.0),
    (4.0, 0.0),
    (3.5, 0.0),
)
LOCAL_TARGET_UNIFORM_RESAMPLE = True
LOCAL_TARGET_UNIFORM_TOLERANCE = 0.35
# Intermediate target cruise altitude is in meters above ground.  AirSim uses
# NED coordinates, so runtime z becomes OCCUPANCY_GROUND_Z - altitude.
LOCAL_TARGET_RANDOMIZE_INTERMEDIATE_ALTITUDE = True
LOCAL_TARGET_INTERMEDIATE_ALTITUDE_RANGE = (20.0, 25.0)
NAVIGATION_START_ALTITUDE = 20.0
NAVIGATION_START_ASCENT_VELOCITY = 3.0
NAVIGATION_START_HOVER_SECONDS = 0.8

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
INTERMEDIATE_STUCK_ACCEPT_DISTANCE = 10.0
INTERMEDIATE_STUCK_PROGRESS_EPS = 0.12
INTERMEDIATE_STUCK_SPEED_MAX = 1.0
INTERMEDIATE_STUCK_WINDOW = 12
INTERMEDIATE_ADVANCE_ACCEPT_DISTANCE = 10.0
INTERMEDIATE_ADVANCE_PROGRESS_RATIO = 0.70
INTERMEDIATE_ADVANCE_MIN_PROGRESS = 3.0
INTERMEDIATE_STALL_MIN_STEPS = 60
INTERMEDIATE_STALL_WINDOW = 18
INTERMEDIATE_STALL_PROGRESS_EPS = 0.06
INTERMEDIATE_STALL_SPEED_MAX = 1.0
SEGMENT_SHIELD_STEP_EXTENSION = 1
SEGMENT_SHIELD_MAX_EXTRA_STEPS = 60

# Runtime safety shield for test-time lower-policy execution.
SAFETY_SHIELD_ENABLED = True
SAFETY_SHIELD_FRONT_WARN_DIST = 4.0
SAFETY_SHIELD_FRONT_STOP_DIST = 2.2
SAFETY_SHIELD_ANY_STOP_DIST = 1.4
SAFETY_SHIELD_DEPTH_WARN_DIST = 4.0
SAFETY_SHIELD_DEPTH_STOP_DIST = 2.0
SAFETY_SHIELD_MIN_BOTTOM_DIST = 2.5
SAFETY_SHIELD_MIN_TOP_DIST = 2.5
SAFETY_SHIELD_EMERGENCY_BOTTOM_DIST = 1.0
SAFETY_SHIELD_EMERGENCY_TOP_DIST = 1.0
SAFETY_SHIELD_CLIMB_ACTION_Z = -0.6
SAFETY_SHIELD_DESCEND_ACTION_Z = 0.5
SAFETY_SHIELD_BRAKE_GAIN = 0.75
SAFETY_SHIELD_WARN_BLEND = 0.35
SAFETY_SHIELD_LOG_EVENTS = True
SAFETY_SHIELD_RECOVERY_ENABLED = True
SAFETY_SHIELD_RECOVERY_STEPS = 6
SAFETY_SHIELD_RECOVERY_BRAKE_STEPS = 2
SAFETY_SHIELD_RECOVERY_LATERAL_ACTION = 0.65
SAFETY_SHIELD_RECOVERY_BACK_ACTION = 0.55
SAFETY_SHIELD_RECOVERY_GOAL_ACTION = 0.25
SAFETY_SHIELD_RECOVERY_CLIMB_ACTION = -0.35
SAFETY_SHIELD_RECOVERY_FRONT_CLEAR_DIST = 4.5
SAFETY_SHIELD_RECOVERY_TRIGGER_REASONS = (
    "emergency_lidar_any",
    "emergency_lidar_front",
    "emergency_depth_front",
    "warn_front",
)

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
VISUALIZE_GLOBAL_PATH = False
VISUAL_Z_OFFSET = 20.0

# Top-down depth-map capture.  Keep this at the exact center of the configured
# occupancy bounds so expanded captures/crops stay symmetric around Astar_planner's
# planning map.
TOPDOWN_CAMERA_CENTER_X = 15.0
TOPDOWN_CAMERA_CENTER_Y = -5.0
