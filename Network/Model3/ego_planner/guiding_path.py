import numpy as np


class GuidingPath:
    """
    Generates a guiding path from the local target provided by the RL policy.
    The guiding path provides reference waypoints for B-spline initialization.

    The local_target[:3] from the RL policy is already in world frame
    (converted by local_target_to_world before calling plan()).
    """

    def __init__(self, horizon=5.0, n_points=10):
        """
        Args:
            horizon: planning horizon in meters
            n_points: number of waypoints in the guiding path
        """
        self.horizon = horizon
        self.n_points = n_points

    def generate(self, current_pos, current_vel, local_target, voxel_grid):
        """
        Generate a guiding path from current position toward local target.

        Args:
            current_pos: (3,) current UAV position
            current_vel: (3,) current UAV velocity
            local_target: (6,) [x, y, z, phi, theta, psi] in world frame
            voxel_grid: VoxelGrid for obstacle avoidance

        Returns:
            waypoints: (n_points, 3) array of guiding waypoints
        """
        # local_target[:3] is already world-frame position
        target_pos = local_target[:3]

        direction = target_pos - current_pos
        dist = np.linalg.norm(direction)
        if dist < 1e-6:
            vel_norm = np.linalg.norm(current_vel)
            if vel_norm > 1e-6:
                direction = current_vel / vel_norm
            else:
                direction = np.array([1.0, 0.0, 0.0])
            dist = self.horizon
        else:
            direction = direction / dist

        waypoints = np.zeros((self.n_points, 3))

        for i in range(self.n_points):
            frac = (i + 1) / self.n_points
            pt = current_pos + frac * direction * min(dist, self.horizon)

            # Push away from obstacles
            obs_dist, obs_pt = voxel_grid.get_distance_with_obstacle(pt, search_radius=1.0)
            if obs_dist < 1.0 and obs_pt is not None:
                away = pt - obs_pt
                away_norm = np.linalg.norm(away)
                if away_norm > 1e-6:
                    push = (1.0 - obs_dist) * away / away_norm
                    pt += push * 0.5

            waypoints[i] = pt

        return waypoints
