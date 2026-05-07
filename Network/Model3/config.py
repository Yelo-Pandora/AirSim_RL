"""
Model3 Configuration
All hyperparameters and constants for RL + EGO-Planner hybrid system.
"""

# === RL Hyperparameters ===
LEARNING_RATE = 1e-3
BUFFER_SIZE = 2 ** 18  # 262144
BATCH_SIZE = 256
GAMMA = 0.99
TAU = 0.005
POLICY_NOISE = 0.2
NOISE_CLIP = 0.5
POLICY_FREQ = 2
TOTAL_TIMESTEPS = 1_500_000
SAVE_FREQ = 50_000

# === EGO-Planner Parameters ===
PB = 3  # B-spline degree (cubic)
NC = 25  # Number of control points
INIT_DT = 0.3  # Initial time interval between control points (seconds)
HORIZON = 7.0  # Planning horizon in meters
PLANNER_DT_SCALE = 0.3  # dt per control point in meters (initial spacing)

# === Cost Function Weights ===
LAMBDA_S = 1.0  # Smoothness weight
LAMBDA_C = 10.0  # Collision weight
LAMBDA_D = 5.0  # Feasibility weight
LAMBDA_F = 20.0  # Curve fitting weight

# === Physical Limits ===
V_MAX = 5.0  # Max velocity (m/s)
A_MAX = 8.0  # Max acceleration (m/s^2)
J_MAX = 20.0  # Max jerk (m/s^3)
V_MARGIN = 0.9  # Elastic coefficient for feasibility (lambda < 1)

# === Collision Avoidance ===
SF = 0.5  # Safety clearance (meters)
TRAJECTORY_RADIUS = 0.5  # Collision check radius for trajectory tube

# === Voxel Grid ===
VOXEL_RES = 0.2  # Voxel resolution (meters)
GRID_HALF_X = 15  # Half grid size in X (total 30m)
GRID_HALF_Y = 15  # Half grid size in Y (total 30m)
GRID_HALF_Z = 5  # Half grid size in Z (total 10m)
MAX_DEPTH = 20.0  # Maximum depth sensor range (meters)
MAX_LIDAR_RANGE = 20.0  # Maximum LiDAR range (meters)

# === A* Parameters ===
ASTAR_HEURISTIC_WEIGHT = 1.0
ASTAR_NEIGHBORHOOD = 26  # 6 or 26 connected

# === Time Reallocation ===
TIME_REALLOCATION_MARGIN = 1.2  # Safety margin for time reallocation

# === Curve Fitting ===
FITTING_AXIS_RATIO = 5.0  # b/a ratio for anisotropic fitting (radial vs axial)
FITTING_A = 1.0  # Semi-major axis (axial direction)
FITTING_B = 0.2  # Semi-minor axis (radial direction)

# === Environment ===
MAX_STEPS = 200  # Max steps per episode
ARRIVAL_RADIUS = 1.5  # Goal arrival radius (meters)
HEIGHT_MIN = -50.0  # Minimum height (NED frame, -50 = 50m up)
HEIGHT_MAX = 0.5  # Maximum height (NED frame, close to ground)
DT_AIRSIM = 0.5  # AirSim step duration (seconds)
REPLAN_EVERY = 5  # Replan every N AirSim steps

# === AirSim ===
AIRSIM_IP = "127.0.0.1"
VEHICLE_NAME_DEFAULT = "Drone1"
TAKEOFF_HEIGHT = 3.0

# === Dataset ===
DATASET_CSV = "relative_coordinates_export.csv"

# === Observation ===
DEPTH_HEIGHT = 9
DEPTH_WIDTH = 16
LIDAR_DIM = 105
KINEMATICS_DIM = 10
PLANNER_INFO_DIM = 4  # traj_length, collision_count, feasibility_ratio, replan_count
