import numpy as np


class GuidingPath:
    """
    Generates a guiding path from the local target provided by the RL policy.
    The guiding path provides reference points for the B-spline initialization.
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
            local_target: (6,) [x, y, z, φ, θ, ψ] from RL policy
            voxel_grid: VoxelGrid for obstacle avoidance

        Returns:
            waypoints: (n_points, 3) array of guiding waypoints
        """
        # Extract position and orientation from local target
        target_pos = local_target[:3]
        target_angles = local_target[3:6]  # φ, θ, ψ

        # Convert from body frame to world frame
        direction = self._body_to_world(current_pos, target_pos, current_vel)

        # Generate intermediate waypoints with obstacle avoidance
        waypoints = np.zeros((self.n_points, 3))
        step = self.horizon / self.n_points

        for i in range(self.n_points):
            frac = (i + 1) / self.n_points
            pt = current_pos + frac * direction * self.horizon

            # Push away from obstacles
            dist, obs_pt = voxel_grid.get_distance_with_obstacle(pt, search_radius=1.0)
            if dist < 1.0 and obs_pt is not None:
                away = pt - obs_pt
                away_norm = np.linalg.norm(away)
                if away_norm > 1e-6:
                    push = (1.0 - dist) * away / away_norm
                    pt += push * 0.5

            waypoints[i] = pt

        # Ensure last waypoint is close to local target
        waypoints[-1] = current_pos + direction * min(np.linalg.norm(direction), self.horizon)

        return waypoints

    def _body_to_world(self, current_pos, body_target, current_vel):
        """Convert body-frame local target to world-frame direction."""
        raw_dir = body_target - current_pos
        dist = np.linalg.norm(raw_dir)
        if dist < 1e-6:
            # Use current velocity direction if too close
            vel_norm = np.linalg.norm(current_vel)
            if vel_norm > 1e-6:
                return current_vel / vel_norm
            return np.array([1.0, 0.0, 0.0])
        return raw_dir / dist
