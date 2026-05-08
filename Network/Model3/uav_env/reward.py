import numpy as np


def compute_reward(current_pos, prev_pos, goal, min_dist_to_obstacle,
                   prev_vel=None, curr_vel=None,
                   sigma=2.0, beta=25.0):
    """
    Compute reward with multiple components inspired by Model2's design philosophy.

    Core components:
      - Distance progress: primary positive feedback for moving toward goal
      - Obstacle penalty: exponential penalty for proximity to obstacles
      - Stagnation penalty: discourages hovering without making progress
      - Speed encouragement: rewards reasonable forward motion
      - Direction alignment: bonus for moving toward goal (not away)
      - Height safety: penalize flying too close to altitude limits

    Args:
        current_pos: (3,) current position
        prev_pos: (3,) previous position
        goal: (3,) global target
        min_dist_to_obstacle: float, minimum distance to nearest obstacle
        prev_vel: (3,) previous velocity (optional)
        curr_vel: (3,) current velocity (optional)
        sigma: obstacle penalty decay rate
        beta: distance reward scale

    Returns:
        reward: float
    """
    prev_dist = np.linalg.norm(prev_pos - goal)
    curr_dist = np.linalg.norm(current_pos - goal)
    dist_delta = prev_dist - curr_dist

    # === Core distance reward (always active) ===
    # Asymmetric: approaching goal is rewarded more than retreating is penalized
    if dist_delta > 0:
        r_dist = dist_delta * beta
    else:
        r_dist = dist_delta * beta * 0.5

    # === Obstacle penalty (always active) ===
    r_obs = -np.exp(-sigma * min_dist_to_obstacle)

    reward = r_dist + r_obs

    # === Stagnation penalty ===
    # If the drone barely moves, penalize to discourage hovering
    if abs(dist_delta) < 0.1:
        reward -= 0.5

    # === Speed encouragement ===
    # Reward moving at reasonable speed (capped to prevent exploitation)
    if curr_vel is not None:
        speed = np.linalg.norm(curr_vel)
        reward += min(speed, 3.0) * 0.2

    # === Smooth acceleration ===
    # Encourage consistent acceleration (penalize erratic velocity changes)
    if curr_vel is not None and prev_vel is not None:
        accel = np.linalg.norm(np.array(curr_vel) - np.array(prev_vel))
        if accel < 2.0:
            reward += 0.1

    # === Direction alignment ===
    # Bonus when velocity direction aligns with goal direction (horizontal plane)
    if curr_vel is not None:
        vel_h = np.array([curr_vel[0], curr_vel[1], 0.0])
        goal_h = np.array([goal[0] - current_pos[0], goal[1] - current_pos[1], 0.0])
        vel_h_norm = np.linalg.norm(vel_h)
        goal_h_norm = np.linalg.norm(goal_h)
        if vel_h_norm > 0.1 and goal_h_norm > 0.5:
            cos_sim = np.dot(vel_h, goal_h) / (vel_h_norm * goal_h_norm)
            reward += cos_sim * 1.0

    # === Height safety ===
    # Penalize flying too close to altitude boundaries
    z = current_pos[2]
    # Too low (close to ground / altitude_min)
    if z < 1.0:
        reward -= 10.0 * (1.0 - z) ** 2
    # Too high (close to altitude_max)
    if z > 8.0:
        reward -= 10.0 * (z - 8.0) ** 2

    return reward
