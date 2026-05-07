import numpy as np
from .bspline import BSpline
from .voxel_grid import VoxelGrid
from .optimizer import TrajectoryOptimizer
from .guiding_path import GuidingPath


class EGOPlanner:
    """
    EGO-Planner interface: receives local target from RL policy,
    generates smooth, collision-free B-spline trajectory.
    """

    def __init__(self, config):
        self.config = config
        self.voxel_grid = VoxelGrid(
            resolution=config.VOXEL_RESOLUTION,
            grid_size=config.VOXEL_GRID_SIZE,
        )
        self.optimizer = TrajectoryOptimizer(config)
        self.guiding_path = GuidingPath(
            horizon=config.PLANNER_HORIZON,
            n_points=config.BSPLINE_CTRL_POINTS,
        )
        self.current_spline = None
        self.spline_start_time = None

    def update_obstacles(self, origin, ray_directions, ray_distances, max_range=10.0):
        """
        Update voxel grid with latest sensor data.

        Args:
            origin: (3,) sensor origin position
            ray_directions: (N, 3) unit vectors for each ray
            ray_distances: (N,) measured distances
            max_range: maximum sensor range
        """
        rays = []
        for i in range(len(ray_directions)):
            d = min(ray_distances[i], max_range)
            direction = np.array(ray_directions[i], dtype=np.float64)
            if np.linalg.norm(direction) > 1e-6:
                direction = direction / np.linalg.norm(direction)
            rays.append((direction, d))
        self.voxel_grid.load_from_rangefinder(origin, rays, max_range)

    def plan(self, current_pos, current_vel, local_target, target_vel=None):
        """
        Plan a trajectory to the local target.

        Args:
            current_pos: (3,) current UAV position
            current_vel: (3,) current UAV velocity
            local_target: (6,) [x, y, z, φ, θ, ψ] from RL
            target_vel: (3,) desired velocity at local target

        Returns:
            BSpline trajectory
        """
        if target_vel is None:
            target_vel = np.zeros(3)

        # Generate guiding path for initialization
        waypoints = self.guiding_path.generate(
            current_pos, current_vel, local_target, self.voxel_grid
        )

        # Use guiding path waypoints as initial control points
        init_ctrl = np.vstack([current_pos.reshape(1, 3), waypoints, local_target[:3].reshape(1, 3)])

        # Optimize trajectory
        spline = self.optimizer.optimize(
            start_pos=current_pos,
            start_vel=current_vel,
            target_pos=local_target[:3],
            target_vel=target_vel,
            voxel_grid=self.voxel_grid,
            init_ctrl_points=init_ctrl,
        )

        self.current_spline = spline
        return spline

    def get_control_command(self, t):
        """
        Get control command at time t along current spline.

        Returns:
            pos: (3,) desired position
            vel: (3,) desired velocity
        """
        if self.current_spline is None:
            return None, None

        pos = self.current_spline.eval(t)
        vel = self.current_spline.eval_derivative(t, order=1)
        return pos, vel

    def check_collision(self, spline=None, radius=0.5):
        """Check if the current trajectory has collisions."""
        if spline is None:
            spline = self.current_spline
        if spline is None:
            return False, 0

        n_samples = 30
        for k in range(n_samples):
            t = k / n_samples * spline.duration
            pt = spline.eval(t)
            dist, _ = self.voxel_grid.get_distance_with_obstacle(pt, search_radius=radius)
            if dist < radius:
                return True, k
        return False, 0
