import numpy as np


def compute_reward(current_pos, prev_pos, goal, min_dist_to_obstacle,
                   sigma=2.0, beta=25.0, current_vel=None, prev_vel=None,
                   action=None, prev_action=None, collided=False,
                   terminal_reason="", safe_altitude=2.0,
                   altitude_weight=4.0, descent_weight=1.0,
                   smooth_action_weight=0.05, speed_smooth_weight=0.02,
                   collision_penalty=25.0, ground_penalty=100.0,
                   out_of_bounds_penalty=50.0, goal_bonus=100.0,
                   return_components=False):
    """
    Compute navigation reward.

    The first two terms keep the RLoPlanner paper's Eq. 6-8 intact:
    r_dist = beta * (||prev_pos - goal|| - ||current_pos - goal||)  (Eq. 6)
    r_obs = -exp(-sigma * d_min)                                     (Eq. 7)

    Additional shaping terms encode practical UAV safety constraints commonly
    used in recent RL navigation work: low-altitude/ground-risk penalty,
    collision/crash terminal penalty, and smoothness penalties.

    Args:
        current_pos: (3,) current position
        prev_pos: (3,) previous position
        goal: (3,) global target
        min_dist_to_obstacle: float, minimum distance to nearest obstacle
        sigma: obstacle penalty decay rate (default 2.0)
        beta: distance reward scale (default 25.0)

    Returns:
        reward: float
    """
    prev_dist = np.linalg.norm(prev_pos - goal)
    curr_dist = np.linalg.norm(current_pos - goal)

    r_dist = beta * (prev_dist - curr_dist)
    r_obs = -np.exp(-sigma * min_dist_to_obstacle)

    altitude = max(-float(current_pos[2]), 0.0)
    prev_altitude = max(-float(prev_pos[2]), 0.0)

    if altitude < safe_altitude:
        low_alt_ratio = (safe_altitude - altitude) / max(safe_altitude, 1e-6)
        r_altitude = -altitude_weight * low_alt_ratio * low_alt_ratio
    else:
        r_altitude = 0.0

    is_descending = altitude < prev_altitude
    if is_descending and altitude < safe_altitude:
        r_descent = -descent_weight * (prev_altitude - altitude)
    else:
        r_descent = 0.0

    if action is not None and prev_action is not None:
        action_delta = np.linalg.norm(np.asarray(action) - np.asarray(prev_action))
        r_action_smooth = -smooth_action_weight * action_delta
    else:
        r_action_smooth = 0.0

    if current_vel is not None and prev_vel is not None:
        speed_delta = np.linalg.norm(np.asarray(current_vel) - np.asarray(prev_vel))
        r_speed_smooth = -speed_smooth_weight * speed_delta
    else:
        r_speed_smooth = 0.0

    r_collision = -collision_penalty if collided else 0.0

    r_terminal = 0.0
    if terminal_reason == "arrived":
        r_terminal += goal_bonus
    elif terminal_reason == "ground":
        r_terminal -= ground_penalty
    elif terminal_reason in ("collision", "out_of_bounds"):
        r_terminal -= out_of_bounds_penalty

    components = {
        "progress": float(r_dist),
        "obstacle": float(r_obs),
        "altitude": float(r_altitude),
        "descent": float(r_descent),
        "action_smooth": float(r_action_smooth),
        "speed_smooth": float(r_speed_smooth),
        "collision": float(r_collision),
        "terminal": float(r_terminal),
    }
    reward = float(sum(components.values()))

    if return_components:
        return reward, components
    return reward
