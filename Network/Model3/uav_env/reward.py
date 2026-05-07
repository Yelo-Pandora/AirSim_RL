import numpy as np


def compute_reward(current_pos, prev_pos, goal, min_dist_to_obstacle, sigma=2.0, beta=25.0):
    """
    Compute reward per paper Eq. 6-8.

    r_dist = β * (||prev_pos - goal|| - ||current_pos - goal||)
    r_obs = -exp(-σ * d_min)
    r = r_dist + r_obs

    Args:
        current_pos: (3,) current position
        prev_pos: (3,) previous position
        goal: (3,) global target
        min_dist_to_obstacle: float, minimum distance to nearest obstacle
        sigma: obstacle penalty decay σ (default 2.0)
        beta: distance reward scale β (default 25.0)

    Returns:
        reward: float
    """
    prev_dist = np.linalg.norm(prev_pos - goal)
    curr_dist = np.linalg.norm(current_pos - goal)

    # Distance reward
    r_dist = beta * (prev_dist - curr_dist)

    # Obstacle penalty
    r_obs = -np.exp(-sigma * min_dist_to_obstacle)

    return r_dist + r_obs
