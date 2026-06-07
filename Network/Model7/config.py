import os
import importlib.util


MODEL7_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.dirname(MODEL7_DIR)
PROJECT_ROOT = os.path.dirname(NETWORK_DIR)

MODEL6_DIR = os.path.join(NETWORK_DIR, "Model6")

_model6_config_path = os.path.join(MODEL6_DIR, "config.py")
_model6_spec = importlib.util.spec_from_file_location("_model6_config", _model6_config_path)
_model6_config = importlib.util.module_from_spec(_model6_spec)
_model6_spec.loader.exec_module(_model6_config)
for _name in dir(_model6_config):
    if _name.isupper():
        globals()[_name] = getattr(_model6_config, _name)

MODEL7_DIR = os.path.dirname(os.path.abspath(__file__))
NETWORK_DIR = os.path.dirname(MODEL7_DIR)
PROJECT_ROOT = os.path.dirname(NETWORK_DIR)
MODEL6_DIR = os.path.join(NETWORK_DIR, "Model6")

DEFAULT_PPO_REPO = os.path.join(
    PROJECT_ROOT,
    "external",
    "PPO-based-Autonomous-Navigation-for-Quadcopters",
)
DEFAULT_PPO_MODEL = os.path.join(
    DEFAULT_PPO_REPO,
    "saved_policy",
    "ppo_navigation_policy.zip",
)

# Lower PPO execution.
PPO_IMAGE_SHAPE = (50, 50, 3)
PPO_FORWARD_SPEED = 0.4
PPO_LATERAL_SPEED = 0.4
PPO_VERTICAL_SPEED = 0.4
PPO_ACTION_DURATION = 1.0
PPO_SEGMENT_MAX_STEPS = 180
PPO_DETERMINISTIC_POLICY = True

SEGMENT_GOAL_TOLERANCE = 2.0
INTERMEDIATE_AXIS_TOLERANCE = 3.0
FINAL_AXIS_TOLERANCE = 1.5
STOP_ON_SEGMENT_FAILURE = True

# Visualization.
VISUALIZE_GLOBAL_PATH = True
VISUAL_Z_OFFSET = 20.0
