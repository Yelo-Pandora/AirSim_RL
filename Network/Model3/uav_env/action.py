import numpy as np


def local_target_to_world(current_pos, current_yaw, action):
    """
    Convert body-frame local target to world-frame position.

    Args:
        current_pos: (3,) UAV world position
        current_yaw: float, UAV yaw angle
        action: (6,) [x, y, z, φ, θ, ψ] local target in body frame

    Returns:
        world_target: (3,) world-frame target position
        target_angles: (3,) target orientation [φ, θ, ψ]
    """
    body_pos = action[:3]
    target_angles = action[3:6]

    # Rotation matrix for yaw (around z-axis)
    cos_yaw = np.cos(current_yaw)
    sin_yaw = np.sin(current_yaw)
    R_z = np.array([
        [cos_yaw, -sin_yaw, 0],
        [sin_yaw,  cos_yaw, 0],
        [0,        0,       1],
    ])

    world_pos = current_pos + R_z @ body_pos
    return world_pos, target_angles


def clip_action(action, pos_min=1.0, pos_max=5.0):
    """
    Clip action to valid range.
    Position part: scaled to [pos_min, pos_max]
    Angle part: [-π, π]
    """
    pos = action[:3].copy()
    angles = action[3:6].copy()

    # Normalize position to desired range
    pos_norm = np.linalg.norm(pos)
    if pos_norm > 1e-6:
        pos = pos / pos_norm * np.clip(pos_norm, pos_min, pos_max)
    else:
        pos = np.array([pos_min, 0.0, 0.0])

    # Clip angles
    angles = np.clip(angles, -np.pi, np.pi)

    return np.concatenate([pos, angles])
