# config.py — RLoPlanner hyperparameters from the paper

# ============== UAV Model ==============
UAV_MAX_SPEED = 2.1          # m/s
UAV_ALTITUDE_MIN = 0.0       # m
UAV_ALTITUDE_MAX = 10.0      # m
UAV_RADIUS = 0.5             # m, collision radius
UAV_ARRIVE_DIST = 0.5        # m, goal arrival threshold

# ============== EGO-Planner ==============
PLANNER_HORIZON = 5.0        # m, trajectory planning horizon
PLANNER_DT = 0.1             # s, control loop period
PLANNER_OPTIM_ITERS = 50     # gradient descent iterations per planning step
PLANNER_LR = 0.1             # learning rate for trajectory optimization

# B-spline parameters
BSPLINE_ORDER = 4            # cubic B-spline
BSPLINE_CTRL_POINTS = 10     # number of control points
BSPLINE_DT = 0.2             # time interval between control points

# Cost function weights (Eq. 17: J = λ1*Js + λ2*Jc + λ3*Jd + λ4*Jlp)
COST_SMOOTH_WEIGHT = 1.0     # λ1, trajectory smoothness
COST_COLLISION_WEIGHT = 10.0  # λ2, obstacle avoidance
COST_DYNAMIC_WEIGHT = 1.0    # λ3, dynamic feasibility
COST_LOCAL_TARGET_WEIGHT = 5.0  # λ4, local target tracking

# Local target cost params (Eq. 18)
LOCAL_TARGET_POS_WEIGHT = 1.0    # λp, position deviation weight
LOCAL_TARGET_VEL_WEIGHT = 0.5    # λv, velocity deviation weight

# Voxel grid
VOXEL_RESOLUTION = 0.2       # m, voxel size
VOXEL_GRID_SIZE = 50         # grid size in each dimension (cells)

# ============== RSPG Algorithm ==============
RSPG_GAMMA = 0.99            # discount factor
RSPG_LR = 0.01               # Adam learning rate
RSPG_TAU = 0.01              # target network soft update rate
RSPG_REPLAY_BUFFER = 100000  # replay buffer capacity
RSPG_BATCH_SIZE = 256        # batch size for sampling
RSPG_MAX_HISTORY = 100       # max LSTM unroll length
RSPG_GRAD_CLIP = 1.0         # gradient clipping norm
RSPG_ENTROPY_ALPHA_INIT = 8.0  # initial entropy coefficient α
RSPG_ALPHA_LR = 0.0003       # learning rate for entropy coefficient
RSPG_TARGET_ENTROPY = -3.0   # target entropy for automatic alpha tuning

# ============== Observation Space ==============
OBS_DIM_IMU = 2              # [yaw, pitch]
OBS_DIM_RANGE = 25           # 5x5 rangefinder grid
OBS_DIM_TARGET = 4           # relative position (3) + distance (1)
OBS_DIM = OBS_DIM_IMU + OBS_DIM_RANGE + OBS_DIM_TARGET  # = 31

# Rangefinder config
RANGE_MAX = 10.0             # m, max rangefinder range
RANGE_HFOV = 90.0            # degrees, horizontal FOV
RANGE_VFOV = 60.0            # degrees, vertical FOV
RANGE_RAYS_H = 5             # horizontal rays
RANGE_RAYS_V = 5             # vertical rays

# ============== Action Space ==============
ACTION_DIM = 6               # [x, y, z, φ, θ, ψ] local target
ACTION_POS_MIN = 1.0         # m, min local target distance
ACTION_POS_MAX = 5.0         # m, max local target distance

# ============== Reward ==============
REWARD_SIGMA = 2.0           # obstacle penalty decay σ
REWARD_BETA = 25.0           # distance reward scale β

# ============== Training ==============
TRAIN_EPISODES = 200         # total training episodes
TRAIN_MAX_STEPS = 500        # max steps per episode
TRAIN_SAVE_INTERVAL = 10     # save model every N episodes
TRAIN_LOG_INTERVAL = 1       # log metrics every N episodes
TRAIN_SEED = 42              # random seed

# ============== Paths ==============
MODEL_SAVE_DIR = "models/saved_models"
LOG_DIR = "logs"
