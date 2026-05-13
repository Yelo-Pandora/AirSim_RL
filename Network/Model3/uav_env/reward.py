import numpy as np


def compute_reward(current_pos, prev_pos, goal, min_dist_to_obstacle,
                   sigma=2.0, beta=25.0):
    """
    Compute reward per paper Eq. 6-8.

    r_dist = beta * (||prev_pos - goal|| - ||current_pos - goal||)  (Eq. 6)
    r_obs = -exp(-sigma * d_min)                                     (Eq. 7)
    r = r_dist + r_obs                                               (Eq. 8)

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

    # Distance reward (Eq. 6)
    r_dist = beta * (prev_dist - curr_dist)

    # Obstacle penalty (Eq. 7)
    r_obs = -np.exp(-sigma * min_dist_to_obstacle)

    return r_dist + r_obs  # (Eq. 8)
