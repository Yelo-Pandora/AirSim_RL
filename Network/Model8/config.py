import importlib.util
import os


MODEL8_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.dirname(MODEL8_DIR)
PROJECT_ROOT = os.path.dirname(NETWORK_DIR)
MODEL6_DIR = os.path.join(NETWORK_DIR, "Model6")

_model6_config_path = os.path.join(MODEL6_DIR, "config.py")
_model6_spec = importlib.util.spec_from_file_location("_model6_config", _model6_config_path)
_model6_config = importlib.util.module_from_spec(_model6_spec)
_model6_spec.loader.exec_module(_model6_config)
for _name in dir(_model6_config):
    if _name.isupper():
        globals()[_name] = getattr(_model6_config, _name)

MODEL8_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.dirname(MODEL8_DIR)
PROJECT_ROOT = os.path.dirname(NETWORK_DIR)
MODEL6_DIR = os.path.join(NETWORK_DIR, "Model6")

DEFAULT_DDPG_REPO = os.path.join(
    PROJECT_ROOT,
    "external",
    "Drone-navigation-and-obstacle-avoidance-using-DDPG",
)
DEFAULT_DDPG_MODEL = os.path.join(
    DEFAULT_DDPG_REPO,
    "Lidar",
    "model",
    "lidar_model.zip",
)

# Lower DDPG execution.  The external lidar DDPG policy expects a 1x5 vector:
# [front_lidar, left_lidar, right_lidar, distance_to_goal, vertical_offset].
DDPG_SEGMENT_MAX_STEPS = 180
DDPG_DETERMINISTIC_POLICY = True
DDPG_ACTION_DURATION = 0.1
DDPG_ACTION_SPEED_SCALE = 0.7
DDPG_ACTION_YAW_SCALE = 2.0
DDPG_MAX_DEPTH = 10.0
DDPG_MAX_VERTICAL_DIFFERENCE = 5.0
DDPG_LIDAR_MAX_DISTANCE = 10.0
DDPG_CRASH_DISTANCE = 1.0
DDPG_GOAL_RADIUS = 5.0

SEGMENT_GOAL_TOLERANCE = 2.0
INTERMEDIATE_AXIS_TOLERANCE = 3.0
FINAL_AXIS_TOLERANCE = 1.5
STOP_ON_SEGMENT_FAILURE = True

# Visualization.
VISUALIZE_GLOBAL_PATH = True
VISUAL_Z_OFFSET = 20.0
