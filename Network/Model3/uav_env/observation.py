import numpy as np


def build_observation(state, goal, ray_directions, ray_distances):
    """
    Build observation vector for RSPG agent.

    obs = [yaw, pitch, D_25, ξ]
    - yaw, pitch: IMU orientation angles (2)
    - D_25: 5x5 rangefinder distances (25)
    - ξ: relative target info (4) = [rel_pos_normalized, distance]

    Args:
        state: dict or tuple with UAV state
               Expected: {position, velocity, orientation, ...}
        goal: (3,) global target position
        ray_directions: (25, 3) ray directions
        ray_distances: (25,) measured distances

    Returns:
        obs: (31,) observation vector
    """
    if isinstance(state, dict):
        pos = state["position"]
        orientation = state.get("orientation", np.zeros(3))
    else:
        pos = np.array(state[:3])
        orientation = np.array(state[3:6]) if len(state) > 3 else np.zeros(3)

    # IMU: yaw and pitch angles
    yaw = orientation[2] if len(orientation) > 2 else 0.0
    pitch = orientation[1] if len(orientation) > 1 else 0.0

    # Normalize angles
    yaw = np.clip(yaw, -np.pi, np.pi)
    pitch = np.clip(pitch, -np.pi / 2, np.pi / 2)

    imu_obs = np.array([yaw, pitch])

    # Rangefinder distances (25)
    range_obs = np.array(ray_distances, dtype=np.float64)
    # Normalize to [0, 1] assuming max range = 10.0
    range_obs = np.clip(range_obs / 10.0, 0.0, 1.0)

    # Relative target information (4)
    rel_pos = goal - pos
    dist_to_goal = np.linalg.norm(rel_pos)
    if dist_to_goal > 1e-6:
        rel_pos_norm = rel_pos / dist_to_goal
    else:
        rel_pos_norm = np.zeros(3)
    # ξ = [dx_norm, dy_norm, dz_norm, distance]
    target_obs = np.concatenate([rel_pos_norm, [np.clip(dist_to_goal / 50.0, 0.0, 1.0)]])

    obs = np.concatenate([imu_obs, range_obs, target_obs])
    return obs.astype(np.float32)


def get_ray_directions(hfov=90.0, vfov=60.0, rays_h=5, rays_v=5, forward=np.array([1, 0, 0])):
    """
    Generate ray directions for simulated rangefinder.

    Args:
        hfov: horizontal field of view in degrees
        vfov: vertical field of view in degrees
        rays_h: number of horizontal rays
        rays_v: number of vertical rays
        forward: forward direction in body frame

    Returns:
        directions: (rays_h * rays_v, 3) array of unit vectors
    """
    hfov_rad = np.radians(hfov)
    vfov_rad = np.radians(vfov)

    h_angles = np.linspace(-hfov_rad / 2, hfov_rad / 2, rays_h)
    v_angles = np.linspace(-vfov_rad / 2, vfov_rad / 2, rays_v)

    directions = []
    for h in h_angles:
        for v in v_angles:
            # Spherical coordinates: forward is x-axis
            dir_x = np.cos(h) * np.cos(v)
            dir_y = np.sin(h) * np.cos(v)
            dir_z = np.sin(v)
            d = np.array([dir_x, dir_y, dir_z])
            d = d / np.linalg.norm(d)
            directions.append(d)

    return np.array(directions)
